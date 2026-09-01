use std::ffi::c_void;
use std::slice;

use super::Binding;

mod sealed {
    pub trait Sealed {}
}

/// Stack-owned DMA binding metadata in declaration order.
pub trait Bindings: sealed::Sealed {
    /// Number of inline device bindings.
    const LEN: usize;

    /// Returns the contiguous FFI binding pointers used to call the runtime.
    fn as_ptrs(&self) -> &[*const c_void];
}

#[repr(C)]
#[derive(Debug, Default)]
pub(crate) struct Empty;

#[repr(C)]
#[derive(Debug)]
pub(crate) struct Concat<L, R> {
    left: L,
    right: R,
}

impl<L, R> Concat<L, R> {
    pub(crate) const fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}

/// Creates empty structural binding metadata.
pub fn empty() -> impl Bindings {
    Empty
}

/// Joins structural binding metadata in declaration order.
pub fn join<L: Bindings, R: Bindings>(left: L, right: R) -> impl Bindings {
    Concat::new(left, right)
}

impl sealed::Sealed for Empty {}

impl Bindings for Empty {
    const LEN: usize = 0;

    fn as_ptrs(&self) -> &[*const c_void] {
        &[]
    }
}

impl sealed::Sealed for Binding {}

impl Bindings for Binding {
    const LEN: usize = 1;

    fn as_ptrs(&self) -> &[*const c_void] {
        ffi_slice(self, std::ptr::from_ref(self).cast(), Self::LEN)
    }
}

impl<L: Bindings, R: Bindings> sealed::Sealed for Concat<L, R> {}

impl<L: Bindings, R: Bindings> Bindings for Concat<L, R> {
    const LEN: usize = L::LEN + R::LEN;

    fn as_ptrs(&self) -> &[*const c_void] {
        ffi_slice(self, std::ptr::from_ref(self).cast(), Self::LEN)
    }
}

impl<L: Bindings, const N: usize> sealed::Sealed for [L; N] {}

impl<L: Bindings, const N: usize> Bindings for [L; N] {
    const LEN: usize = L::LEN * N;

    fn as_ptrs(&self) -> &[*const c_void] {
        ffi_slice(self, std::ptr::from_ref(self).cast(), Self::LEN)
    }
}

fn ffi_slice<T: ?Sized>(_: &T, ptr: *const Binding, len: usize) -> &[*const c_void] {
    if len == 0 {
        return &[];
    }
    // SAFETY: `Binding` is one transparent pointer. Sealed nonempty `Bindings` contain exactly
    // `len` contiguous, aligned `Binding` leaves that stay live for this borrow.
    unsafe { slice::from_raw_parts(ptr.cast::<*const c_void>(), len) }
}

fn as_ptrs(bindings: &[Binding]) -> &[*const c_void] {
    ffi_slice(bindings, bindings.as_ptr(), bindings.len())
}

/// Runs a kernel with structural binding metadata.
pub async fn run(kernel: &super::Kernel, inputs: &impl Bindings, outputs: &impl Bindings) {
    kernel.run(inputs.as_ptrs(), outputs.as_ptrs()).await;
}

/// Runs a kernel with allocated output bindings.
pub async fn run_into(kernel: &super::Kernel, inputs: &impl Bindings, outputs: &[Binding]) {
    kernel.run(inputs.as_ptrs(), as_ptrs(outputs)).await;
}

#[cfg(test)]
mod tests {
    use std::mem::{ManuallyDrop, align_of, size_of};

    use super::{Bindings, Concat, Empty, join};
    use crate::backend::npu::Binding;

    #[unsafe(no_mangle)]
    extern "C" fn furiosa_npu_buffer_free(_binding: *mut std::ffi::c_void) {}

    fn binding(tag: &u8) -> Binding {
        Binding::from_raw(std::ptr::from_ref(tag).cast_mut().cast())
    }

    #[test]
    fn bindings_are_contiguous() {
        let first = 1;
        let second = 2;
        let third = 3;
        let bindings = ManuallyDrop::new(join(Empty, join(binding(&first), [binding(&second), binding(&third)])));

        assert_eq!(<Concat<Empty, Concat<Binding, [Binding; 2]>> as Bindings>::LEN, 3);
        assert_eq!(
            bindings.as_ptrs(),
            [
                std::ptr::from_ref(&first).cast(),
                std::ptr::from_ref(&second).cast(),
                std::ptr::from_ref(&third).cast(),
            ],
        );
    }

    #[test]
    fn bindings_have_pointer_layout() {
        type Nested = Concat<Empty, Concat<Binding, Empty>>;

        assert_eq!(size_of::<Binding>(), size_of::<*mut ()>());
        assert_eq!(align_of::<Binding>(), align_of::<*mut ()>());
        assert_eq!(size_of::<Nested>(), size_of::<Binding>());
        assert_eq!(align_of::<Nested>(), align_of::<Binding>());
        assert_eq!(size_of::<[Binding; 3]>(), 3 * size_of::<Binding>());
    }
}
