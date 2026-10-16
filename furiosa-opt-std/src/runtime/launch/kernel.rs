use std::marker::PhantomData;

use furiosa_opt_rt::registry::Key;

use super::{Launched, Returns, Submitted};
use crate::Error;
use crate::backend::Runtime;
use crate::runtime::{CurrentBackend, Device, DeviceFn, DeviceSend};

/// A function a `#[device]` item declares, which the selected backend finds and loads. A generic
/// function loads the instantiation its arguments `Args` select.
pub trait Loadable<Args: DeviceSend>: DeviceFn<Args> + Copy {
    /// Finds the function and loads it on `device`.
    fn load(self, device: &Device) -> Result<Kernel<Self>, Error>;
    /// Submits work through `kernel`, which this function loaded.
    #[doc(hidden)]
    fn run<R: Returns<Self::Output>>(kernel: Kernel<Self>, args: Args, returns: R) -> Submitted<Launched<R::Value>>;
}

/// A device function found and loaded once: its launches skip the lookup, and its persistent
/// allocations exist before a caller sizes its own. A device it was not loaded on loads it at its
/// first launch there, on the calling thread. It launches only the instantiation it loaded; other
/// arguments of a generic function fail with [`Error::Invalid`].
#[derive(Clone, Copy, Debug)]
pub struct Kernel<F> {
    function: PhantomData<F>,
    pub(crate) id: <CurrentBackend as Runtime>::Kernel,
    instance: u64,
}

impl<F> Kernel<F> {
    pub(crate) fn new(_: F, id: <CurrentBackend as Runtime>::Kernel, key: Key<'_>) -> Self {
        Self {
            function: PhantomData,
            id,
            instance: Self::instance(key),
        }
    }

    pub(crate) fn of(self, key: Key<'_>) -> Result<Self, Error> {
        (self.instance == Self::instance(key))
            .then_some(self)
            .ok_or(Error::Invalid(
                "the kernel was loaded for another instantiation of the function",
            ))
    }

    fn instance(key: Key<'_>) -> u64 {
        let fold = |hash: u64, value: u64| (hash ^ value).wrapping_mul(0x0100_0000_01b3);
        match key {
            Key::Numeric(key) => key.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &value| {
                fold(
                    hash,
                    u64::try_from(value).unwrap_or_else(|_| unreachable!("registry keys fit u64")),
                )
            }),
            Key::Type(key) => key
                .bytes()
                .fold(0xcbf2_9ce4_8422_2324, |hash, byte| fold(hash, u64::from(byte))),
        }
    }
}

impl<F: Loadable<Args>, Args: DeviceSend> DeviceFn<Args> for Kernel<F> {
    type Output = F::Output;

    fn launch<R: Returns<F::Output>>(self, args: Args, returns: R) -> Submitted<Launched<R::Value>> {
        F::run(self, args, returns)
    }
}
