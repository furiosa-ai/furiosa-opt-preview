//! Memory-mapped registers. Naming one is the only unsafe step; from then on it is read and
//! written like a cell. A hardware unit lists its registers as constants of this type, so the
//! `unsafe` that says "this is a register, mapped for as long as I run" appears once per register
//! and nowhere in the unit's logic. The host names registers the same way through a mapping.

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
// one peripheral, so every hand-off between memory and a register has a barrier.

/// Orders register reads before the memory reads that follow.
pub fn acquire() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier alone.
    unsafe {
        core::arch::asm!("dmb oshld", options(nostack, preserves_flags))
    };
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
}

/// Orders memory writes before the register write that follows.
pub fn release() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier alone.
    unsafe {
        core::arch::asm!("dmb oshst", options(nostack, preserves_flags))
    };
    // Only `SeqCst` emits a fence instruction on x86 (`mfence`), which write-combined stores to
    // mapped device memory need; `Release` orders only the compiler there.
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Orders memory reads and writes before the register write that follows, so a producer cannot
/// overwrite an entry still being read.
pub(crate) fn fence() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier alone.
    unsafe {
        core::arch::asm!("dmb osh", options(nostack, preserves_flags))
    };
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Completes every memory access before the register write that follows, for an engine that
/// fetches its descriptors from memory.
pub fn complete() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: a barrier alone.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags))
    };
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}
