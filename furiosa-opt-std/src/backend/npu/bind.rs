use furiosa_mapping::M;

use super::Binding;
use super::ffi::{furiosa_npu_buffer_from, rt};
use crate::backend::Backend;
use crate::runtime::DeviceSend;
use crate::scalar::Scalar;
use crate::tensor::memory::{Address, HbmTensor, HbmTensorView, HbmTensorViewMut};

fn to_binding(address: Option<Address>, len: usize) -> Binding {
    // Cpu and compiler-placed tensors have no host-visible address. Such a value cannot cross this
    // adapter, which is reached only when a host NPU launch binds an existing HBM allocation.
    let Some(address) = address else {
        panic!("NPU launch arguments must name a host-placed HBM allocation");
    };
    Binding::from_raw(unsafe { furiosa_npu_buffer_from(rt(), address, len) })
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> From<&HbmTensor<D, Chip, Element, B>> for Binding {
    fn from(tensor: &HbmTensor<D, Chip, Element, B>) -> Self {
        to_binding(tensor.address(), HbmTensor::<D, Chip, Element, B>::size())
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> From<&HbmTensorView<'_, D, Chip, Element, B>> for Binding {
    fn from(view: &HbmTensorView<'_, D, Chip, Element, B>) -> Self {
        to_binding(view.address(), view.addressable_len())
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> From<&HbmTensorViewMut<'_, D, Chip, Element, B>> for Binding {
    fn from(view: &HbmTensorViewMut<'_, D, Chip, Element, B>) -> Self {
        to_binding(view.address(), view.addressable_len())
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensor<D, Chip, Element, B> {
    fn bind(&self) -> impl crate::__private::Bindings {
        Binding::from(self)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensorView<'_, D, Chip, Element, B> {
    fn bind(&self) -> impl crate::__private::Bindings {
        Binding::from(self)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend for HbmTensorViewMut<'_, D, Chip, Element, B> {
    fn bind(&self) -> impl crate::__private::Bindings {
        Binding::from(self)
    }
}

impl DeviceSend for () {
    fn bind(&self) -> impl crate::__private::Bindings {
        crate::__private::empty()
    }
}

impl<T> DeviceSend for std::marker::PhantomData<T> {
    fn bind(&self) -> impl crate::__private::Bindings {
        crate::__private::empty()
    }
}

impl<T: DeviceSend + ?Sized> DeviceSend for &T {
    fn bind(&self) -> impl crate::__private::Bindings {
        T::bind(self)
    }
}

impl<T: DeviceSend + ?Sized> DeviceSend for &mut T {
    fn bind(&self) -> impl crate::__private::Bindings {
        T::bind(self)
    }
}

impl<T: DeviceSend, const N: usize> DeviceSend for [T; N] {
    fn bind(&self) -> impl crate::__private::Bindings {
        std::array::from_fn::<_, N, _>(|index| self[index].bind())
    }
}

macro_rules! concat_value {
    () => { crate::__private::empty() };
    ($head:ident $(, $tail:ident)*) => {
        crate::__private::join($head.bind(), concat_value!($($tail),*))
    };
}

macro_rules! impl_device_send_tuple {
    ($($ty:ident : $value:ident),+ $(,)?) => {
        impl<$($ty: DeviceSend),+> DeviceSend for ($($ty,)+) {
            fn bind(&self) -> impl crate::__private::Bindings {
                let ($($value,)+) = self;
                concat_value!($($value),+)
            }
        }
    };
}

impl_device_send_tuple!(T0: t0);
impl_device_send_tuple!(T0: t0, T1: t1);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21, T22: t22);
impl_device_send_tuple!(T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7, T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15, T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21, T22: t22, T23: t23);
