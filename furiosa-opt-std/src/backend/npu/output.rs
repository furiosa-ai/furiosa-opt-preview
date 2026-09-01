use furiosa_mapping::M;

use super::ffi::furiosa_npu_buffer_offset;
use super::{Binding, Kernel};
use crate::runtime::DeviceSend;
use crate::scalar::Scalar;
use crate::tensor::Tensor;
use crate::tensor::memory::HbmTensor;

impl<D: Scalar, Chip: M, Element: M> From<Binding> for HbmTensor<D, Chip, Element> {
    fn from(binding: Binding) -> Self {
        let addr = unsafe { furiosa_npu_buffer_offset(binding.as_ptr()) };
        Self::from_parts(Tensor::zeroed(), Some(addr)).owns(binding)
    }
}

/// Reconstructs a device-function return value from output bindings in declaration order.
pub trait DeviceOutput: DeviceSend + Sized {
    /// Number of leaf output bindings this value owns.
    fn count() -> usize;
    /// Allocates this value's output bindings on `kernel`.
    fn alloc_into(kernel: &Kernel, bindings: &mut Vec<Binding>);
    /// Rebuilds the value by taking bindings in output order.
    fn take(bindings: &mut impl Iterator<Item = Binding>) -> Self;

    /// Allocates all output bindings.
    fn alloc(kernel: &Kernel) -> Vec<Binding> {
        let mut bindings = Vec::with_capacity(Self::count());
        Self::alloc_into(kernel, &mut bindings);
        bindings
    }
}

impl DeviceOutput for () {
    fn count() -> usize {
        0
    }

    fn alloc_into(_: &Kernel, _: &mut Vec<Binding>) {}

    fn take(_: &mut impl Iterator<Item = Binding>) {}
}

impl<D: Scalar, Chip: M, Element: M> DeviceOutput for HbmTensor<D, Chip, Element> {
    fn count() -> usize {
        1
    }

    fn alloc_into(kernel: &Kernel, bindings: &mut Vec<Binding>) {
        bindings.push(kernel.alloc_binding(Self::size()));
    }

    fn take(bindings: &mut impl Iterator<Item = Binding>) -> Self {
        bindings.next().expect("HbmTensor output is missing its binding").into()
    }
}

macro_rules! impl_device_output_tuple {
    () => {};
    (($T0:ident, $t0:ident) $(, ($T:ident, $t:ident))*) => {
        impl<$T0 $(, $T)*> DeviceOutput for ($T0, $($T,)*)
        where
            $T0: DeviceOutput $(, $T: DeviceOutput)*,
        {
            fn count() -> usize {
                $T0::count() $(+ $T::count())*
            }

            fn alloc_into(kernel: &Kernel, bindings: &mut Vec<Binding>) {
                $T0::alloc_into(kernel, bindings);
                $( $T::alloc_into(kernel, bindings); )*
            }

            fn take(bindings: &mut impl Iterator<Item = Binding>) -> Self {
                ($T0::take(bindings), $($T::take(bindings),)*)
            }
        }

        impl_device_output_tuple!($(($T, $t)),*);
    };
}

impl_device_output_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7),
    (T8, t8),
    (T9, t9),
    (T10, t10),
    (T11, t11),
    (T12, t12),
    (T13, t13),
    (T14, t14),
    (T15, t15),
    (T16, t16),
    (T17, t17),
    (T18, t18),
    (T19, t19),
    (T20, t20),
    (T21, t21),
    (T22, t22),
    (T23, t23)
);
