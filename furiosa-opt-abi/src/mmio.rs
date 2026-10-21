//! Memory-mapped registers. Naming one is the only unsafe step; from then on it is read and
//! written like a cell. A hardware unit lists its registers as constants of this type, so the
//! `unsafe` that says "this is a register, mapped for as long as I run" appears once per register
//! and nowhere in the unit's logic. The host names registers the same way through a mapping.

use core::ops::Range;
use core::ptr;

/// One register of `T`, accessed volatile.
#[derive(Clone, Copy, Debug)]
pub struct Register<T>(*mut T);

// SAFETY: a register has no memory of its own to race on; the hardware serializes accesses.
unsafe impl<T> Sync for Register<T> {}
// SAFETY: as above; a register may be used from whichever thread holds its owner.
unsafe impl<T> Send for Register<T> {}

impl<T: Copy> Register<T> {
    /// # Safety
    ///
    /// `register` must point at a `T`-sized register mapped for as long as the value is used.
    pub const unsafe fn new(register: *mut T) -> Self {
        Self(register)
    }

    /// # Safety
    ///
    /// `address` must be a `T`-sized register the MMU maps for as long as the image runs.
    pub const unsafe fn at(address: usize) -> Self {
        Self(ptr::without_provenance_mut(address))
    }

    pub fn read(&self) -> T {
        // SAFETY: construction (`new` or `at`) promised a mapped register of this width.
        unsafe { ptr::read_volatile(self.0) }
    }

    pub fn write(&self, value: T) {
        // SAFETY: construction (`new` or `at`) promised a mapped register of this width.
        unsafe { ptr::write_volatile(self.0, value) }
    }

    /// The register `index` entries of `T` past this one: a row of a register array.
    ///
    /// # Safety
    ///
    /// The array must extend that far.
    pub const unsafe fn offset(&self, index: usize) -> Self {
        // SAFETY: the caller promised the array reaches `index`.
        Self(unsafe { self.0.add(index) })
    }
}

// On the device, registers and rings are Device memory, which the core keeps in order only within
// one peripheral, so every hand-off between memory and a register goes through one of these.
impl<T: Copy> Register<T> {
    /// Reads the register; the memory reads after it see what the value announced
    /// (load-acquire): a consumer reads the producer index, then the entries it covers.
    pub fn read_acquire(&self) -> T {
        let value = self.read();
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dmb oshld", options(nostack, preserves_flags))
        };
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        value
    }

    /// Writes the register after the memory reads and writes before it (store-release): a
    /// producer fills entries, then moves the index that publishes them; a consumer reads them,
    /// then moves the index that frees them.
    pub fn write_release(&self, value: T) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dmb osh", options(nostack, preserves_flags))
        };
        // Only `SeqCst` emits a fence instruction on x86 (`mfence`), which write-combined stores
        // to mapped device memory need; `Release` orders only the compiler there.
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        self.write(value);
    }

    /// Writes the register once every memory access before it has completed (after a data
    /// synchronization barrier), for an engine that fetches what it works on from memory.
    pub fn write_synchronized(&self, value: T) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: a barrier alone.
        unsafe {
            core::arch::asm!("dsb sy", options(nostack, preserves_flags))
        };
        #[cfg(not(target_arch = "aarch64"))]
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        self.write(value);
    }
}

/// Memory another party shares, as offsets from an origin, mapped for as long as the value lives.
/// A ring or doorbell set owns the memory it lies in, so it cannot outlive it.
///
/// # Safety
///
/// Every offset in [`Memory::mapped`] from [`Memory::origin`] addresses mapped memory, and stays
/// mapped while the value lives.
pub unsafe trait Memory {
    /// Where offset zero lies; it need not be mapped itself.
    fn origin(&self) -> *mut u8;
    /// The offsets that are mapped.
    fn mapped(&self) -> Range<usize>;
}

/// Memory at a fixed address, mapped for as long as the program runs: a core's own registers and
/// the windows through which it sees its peers.
#[derive(Clone, Copy, Debug)]
pub struct Fixed {
    origin: *mut u8,
    start: usize,
    end: usize,
}

// SAFETY: the memory is fixed and mapped for the program's lifetime; the hardware serializes
// accesses to it.
unsafe impl Send for Fixed {}
// SAFETY: as above.
unsafe impl Sync for Fixed {}

impl Fixed {
    /// # Safety
    ///
    /// The offsets `mapped` from `origin` must stay mapped for as long as the program runs.
    pub const unsafe fn new(origin: *mut u8, mapped: Range<usize>) -> Self {
        Self {
            origin,
            start: mapped.start,
            end: mapped.end,
        }
    }
}

// SAFETY: `new` promised the range stays mapped.
unsafe impl Memory for Fixed {
    fn origin(&self) -> *mut u8 {
        self.origin
    }

    fn mapped(&self) -> Range<usize> {
        self.start..self.end
    }
}

// SAFETY: the shared memory lives as long as any of its handles.
#[cfg(feature = "alloc")]
unsafe impl<M: Memory> Memory for alloc::sync::Arc<M> {
    fn origin(&self) -> *mut u8 {
        (**self).origin()
    }

    fn mapped(&self) -> Range<usize> {
        (**self).mapped()
    }
}
