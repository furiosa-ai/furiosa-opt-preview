use std::fmt::{self, Debug};
use std::marker::PhantomData;

use furiosa_opt_rt::registry::Key;

use super::{Launched, Returns, Submitted};
use crate::Error;
use crate::backend::Runtime;
use crate::runtime::{CurrentBackend, Device, DeviceFn, DeviceSend, Topology};

/// A function a `#[device]` item declares, which the selected backend finds and loads. A generic
/// function loads the instantiation its arguments `Args` select.
pub trait Loadable<Args: DeviceSend>: DeviceFn<Args> + Copy {
    /// The function's registry path.
    #[doc(hidden)]
    const PATH: &'static str;
    /// The instantiation's registry key: its generic arguments' values.
    #[doc(hidden)]
    const KEY: &'static [usize];

    /// The instantiation's function-item type name, which tells apart the instantiations of a
    /// function generic over types other than axes; empty when [`Self::KEY`] does. A load calls
    /// it, so the compiler finds every instantiation a program loads, also one it never launches.
    #[doc(hidden)]
    fn instance() -> &'static str;

    /// Finds the function and loads it on `device`.
    fn load(self, device: &Device) -> Result<Kernel<Self, Args>, Error> {
        let key = match Self::instance() {
            "" => Key::Numeric(Self::KEY),
            instance => Key::Type(instance),
        };
        Ok(Kernel {
            function: PhantomData,
            id: CurrentBackend::load(device.pdma.device(), Self::PATH, key)?,
        })
    }

    /// Submits work through `kernel`, which this function loaded.
    #[doc(hidden)]
    fn run<R: Returns<Self::Output>>(
        kernel: Kernel<Self, Args>,
        args: Args,
        returns: R,
    ) -> Submitted<Launched<R::Value>>;
}

/// A device function found and loaded once for its arguments `Args`: its launches skip the lookup,
/// and its persistent allocations exist before a caller sizes its own. A device it was not loaded
/// on loads it at its first launch there, on the calling thread.
///
/// It launches only with `Args`. The arguments of another instantiation do not compile, even ones
/// whose axes have the same sizes, so load the function once for each instantiation:
///
/// ```compile_fail,E0271
/// # #![feature(register_tool)]
/// # #![register_tool(furiosa_opt)]
/// use furiosa_opt_std::prelude::*;
///
/// axes![A = 8, B = 16];
///
/// #[device(chip = 1)]
/// fn forward<X: AxisName>(device: &mut Device, model: &HbmTensor<f32, m![1], m![X]>) {}
///
/// fn main() -> Result<(), Error> {
///     let mut device = Device::new(forward.topology())?;
///     let kernel = Loadable::<(&mut Device, &HbmTensor<f32, m![1], m![A]>)>::load(forward, &device)?;
///     let other = HbmTensor::<f32, m![1], m![B]>::alloc(&device)?;
///     let _ = device.launch(kernel, (&other,));
///     Ok(())
/// }
/// ```
pub struct Kernel<F, Args> {
    /// A kernel holds none of its arguments: `fn() -> _` keeps it covariant, so one kernel takes
    /// fresh borrows at each launch, and `Send` and `Sync` whatever they borrow.
    function: PhantomData<fn() -> (F, Args)>,
    pub(crate) id: <CurrentBackend as Runtime>::Kernel,
}

// By hand: a derive would bound `Args`, and its leading `&mut Device` is not `Copy`.
impl<F, Args> Clone for Kernel<F, Args> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, Args> Copy for Kernel<F, Args> {}

impl<F, Args> Debug for Kernel<F, Args> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Kernel").field("id", &self.id).finish()
    }
}

impl<F: Loadable<Args>, Args: DeviceSend> DeviceFn<Args> for Kernel<F, Args> {
    const TOPOLOGY: Topology = F::TOPOLOGY;
    type Output = F::Output;

    fn launch<R: Returns<F::Output>>(self, args: Args, returns: R) -> Submitted<Launched<R::Value>> {
        F::run(self, args, returns)
    }
}
