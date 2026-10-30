//! What a launch binds: each argument kind's device memory and the access a launch may use on it,
//! and the returns it writes.

use furiosa_mapping::M;

use crate::Error;
use crate::backend::{Backend, Runtime};
use crate::context::{Dma, DmaContext, Tu, TuContext};
use crate::runtime::{CurrentBackend, Device, HbmBufferSlice, Region};
use crate::scalar::{RuntimeScalar, Scalar};
use crate::tensor::memory::{AllChips, DmTensor, HbmScalar, HbmTensor, HbmTensorView, HbmTensorViewMut};

pub use furiosa_opt_rt::Access;

/// A device function's argument: a tensor or view, a reference to one, or tuples, arrays, and
/// derived structs of them. `&T` binds read access; `&mut T` and an owned value bind the access
/// they are given, so a launch passes [`Access::Write`] and only `&` lowers it.
pub trait DeviceSend<B: Runtime = CurrentBackend> {
    /// Hands `bind` this value's device memory in declaration order, each slice on every chip of
    /// its buffer, with `access` where `Self` permits writing and read access elsewhere. Fails on
    /// a `DmTensor` that holds no resident SRAM, and stops at the first slice `bind` refuses.
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error>;
}

/// What a value's device memory goes to, one slice at a time: a callback, so a nested value names
/// no iterator type of its own and binding holds no list of its slices.
pub type Bind<'a, 'b, B = CurrentBackend> = dyn FnMut(HbmBufferSlice<'a, B>, Access) -> Result<(), Error> + 'b;

/// A device function's return value: `()`, an [`HbmTensor`], or a tuple of them.
pub trait DeviceOutput<B: Runtime = CurrentBackend>: DeviceSend<B> + sealed::Sealed {
    /// New memory on `device` for every tensor of the value.
    #[doc(hidden)]
    fn alloc(device: &Device<B>) -> Result<Self, Error>
    where
        Self: Sized;
}

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for () {}

impl<B: Backend> DeviceOutput<B> for () {
    fn alloc(_: &Device<B>) -> Result<(), Error> {
        Ok(())
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> sealed::Sealed for HbmTensor<D, Chip, Element, B> {}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceOutput<B> for HbmTensor<D, Chip, Element, B> {
    /// A function may write what it returns from another chip, so a return takes shared DRAM.
    fn alloc(device: &Device<B>) -> Result<Self, Error> {
        Self::alloc_in(device.pdma.device(), crate::runtime::Kind::DramShared)
    }
}

macro_rules! impl_device_output_tuple {
    () => {};
    ($T0:ident $(, $T:ident)*) => {
        impl<$T0, $($T),*> sealed::Sealed for ($T0, $($T,)*) {}

        impl<B: Backend, $T0: DeviceOutput<B> $(, $T: DeviceOutput<B>)*> DeviceOutput<B> for ($T0, $($T,)*) {
            fn alloc(device: &Device<B>) -> Result<Self, Error> {
                Ok(($T0::alloc(device)?, $($T::alloc(device)?,)*))
            }
        }

        impl_device_output_tuple!($($T),*);
    };
}

impl_device_output_tuple!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23
);

/// Hands `bind` the memory a launch takes, `args`' then `output`'s, with the access each permits;
/// a device binds every chip of it, so an argument that names only some chips of its buffer is
/// [`Error::Invalid`].
pub(crate) fn launch_memory<'a, B: Backend>(
    args: &'a impl DeviceSend<B>,
    output: &'a impl DeviceOutput<B>,
    bind: &mut Bind<'a, '_, B>,
) -> Result<(), Error> {
    let mut whole = |slice: HbmBufferSlice<'a, B>, access| {
        if slice.region.chips != Region::whole(slice.buffer).chips {
            return Err(Error::Invalid("a launch argument must span every chip of its buffer"));
        }
        bind(slice, access)
    };
    args.arguments(Access::Write, &mut whole)?;
    output.arguments(Access::Write, &mut whole)
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend<B> for HbmTensor<D, Chip, Element, B> {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        bind(self.memory(), access)
    }
}

impl<D: RuntimeScalar, B: Backend> DeviceSend<B> for HbmScalar<D, B> {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        bind(self.memory(), access)
    }
}

impl<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M, B: Backend> DeviceSend<B>
    for DmTensor<D, Chip, Cluster, Slice, Element, B>
{
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        bind(HbmBufferSlice::whole(self.owner().ok_or(Error::Unplaced)?), access)
    }
}

/// A shared view only reads, whatever access it is given.
impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend<B> for HbmTensorView<'_, D, Chip, Element, B, AllChips> {
    fn arguments<'a>(&'a self, _: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        bind(self.memory(), Access::Read)
    }
}

impl<D: Scalar, Chip: M, Element: M, B: Backend> DeviceSend<B> for HbmTensorViewMut<'_, D, Chip, Element, B, AllChips> {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        bind(self.memory(), access)
    }
}

impl<B: Backend> DeviceSend<B> for () {
    fn arguments<'a>(&'a self, _: Access, _: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        Ok(())
    }
}

impl<T, B: Backend> DeviceSend<B> for std::marker::PhantomData<T> {
    fn arguments<'a>(&'a self, _: Access, _: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        Ok(())
    }
}

impl<B: Backend> DeviceSend<B> for Device<B> {
    fn arguments<'a>(&'a self, _: Access, _: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        Ok(())
    }
}

impl<const T: Tu, B: Backend> DeviceSend<B> for TuContext<T> {
    fn arguments<'a>(&'a self, _: Access, _: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        Ok(())
    }
}

impl<const DMA: Dma, B: Backend> DeviceSend<B> for DmaContext<DMA, B> {
    fn arguments<'a>(&'a self, _: Access, _: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        Ok(())
    }
}

// A reference binds with the weaker of its kind's access and the access it is given.
impl<B: Backend, T: ?Sized + DeviceSend<B>> DeviceSend<B> for &T {
    fn arguments<'a>(&'a self, _: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        T::arguments(self, Access::Read, bind)
    }
}

impl<B: Backend, T: ?Sized + DeviceSend<B>> DeviceSend<B> for &mut T {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        T::arguments(self, access, bind)
    }
}

impl<B: Backend, T: DeviceSend<B>, const N: usize> DeviceSend<B> for [T; N] {
    fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
        self.iter().try_for_each(|value| value.arguments(access, bind))
    }
}

macro_rules! impl_device_send_tuple {
    () => {};
    ($first:ident : $head:ident $(, $ty:ident : $value:ident)*) => {
        impl<B: Backend, $first: DeviceSend<B>, $($ty: DeviceSend<B>),*> DeviceSend<B> for ($first, $($ty,)*) {
            fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_, B>) -> Result<(), Error> {
                let ($head, $($value,)*) = self;
                $head.arguments(access, bind)?;
                $($value.arguments(access, bind)?;)*
                Ok(())
            }
        }
        impl_device_send_tuple!($($ty:$value),*);
    };
}
impl_device_send_tuple!(
    T0: t0, T1: t1, T2: t2, T3: t3, T4: t4, T5: t5, T6: t6, T7: t7,
    T8: t8, T9: t9, T10: t10, T11: t11, T12: t12, T13: t13, T14: t14, T15: t15,
    T16: t16, T17: t17, T18: t18, T19: t19, T20: t20, T21: t21, T22: t22, T23: t23
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{HbmBuffer, Topology};

    /// One buffer, every chip of it.
    struct Value(HbmBuffer);

    impl DeviceSend for Value {
        fn arguments<'a>(&'a self, access: Access, bind: &mut Bind<'a, '_>) -> Result<(), Error> {
            bind(self.0.slice(.., ..)?, access)
        }
    }

    /// `&T` binds read access, and `&mut T` and an owned value the access they are given.
    #[test]
    #[cfg(not(backend = "npu"))]
    fn references_lower_access() {
        let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let value = || Value(HbmBuffer::alloc(&device, 1).unwrap());
        let (shared, mut exclusive, owned, mut last) = (value(), value(), value(), value());
        let mut nested = &mut exclusive;
        let arguments = (&shared, &mut nested, owned, &&mut last);
        let mut writes = Vec::new();
        arguments
            .arguments(Access::Write, &mut |_, access| {
                writes.push(access == Access::Write);
                Ok(())
            })
            .unwrap();
        assert_eq!(writes, [false, true, true, false]);
    }
}
