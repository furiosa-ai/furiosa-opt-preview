//! Launches and host transfers: prepared, submitted, and awaited.

use std::fmt::{self, Debug};
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::task::{Context, Poll, Waker, ready};
use tracing::Instrument;

use crate::Error;
use crate::runtime::{Device, DeviceError, DeviceFn, DeviceOutput, DeviceSend};

mod kernel;
pub use kernel::{Kernel, Loadable};

/// Where a launch writes its function's returns: new memory it yields ([`Returned`]), or an
/// existing value's (`&mut` the value).
#[doc(hidden)]
pub trait Returns<O: DeviceOutput> {
    /// What the launch yields once it completes.
    type Value;
    /// Submits `run` into this destination's memory on `device`.
    fn launch(
        self,
        device: &mut Device,
        run: impl FnOnce(&mut Device, &mut O) -> Submitted<Launched<()>>,
    ) -> Submitted<Launched<Self::Value>>;
}

impl<O: DeviceOutput> Returns<O> for Returned {
    type Value = O;

    fn launch(
        self,
        device: &mut Device,
        run: impl FnOnce(&mut Device, &mut O) -> Submitted<Launched<()>>,
    ) -> Submitted<Launched<O>> {
        match O::alloc(device) {
            Ok(mut output) => run(device, &mut output).returning(output),
            Err(error) => Submitted(Launched::Failed(error)),
        }
    }
}

impl<O: DeviceOutput> Returns<O> for &mut O {
    type Value = ();

    fn launch(
        self,
        device: &mut Device,
        run: impl FnOnce(&mut Device, &mut O) -> Submitted<Launched<()>>,
    ) -> Submitted<Launched<()>> {
        run(device, self)
    }
}

/// Work that has not started: a launch or a host transfer. `.await` runs it; `.submit()` starts it
/// now and returns it as [`Submitted`] to await later.
#[must_use = "work does nothing until submitted or awaited"]
pub struct Prepared<W>(W);

/// What a [`Prepared`] starts.
#[doc(hidden)]
pub trait Work {
    /// What the work yields once it completes.
    type Output;
    /// The started work.
    type Future: Future<Output = Result<Self::Output, Error>>;
    /// Starts the work.
    fn submit(self) -> Submitted<Self::Future>;
}

impl<W: Work> Prepared<W> {
    pub(crate) fn new(work: W) -> Self {
        Self(work)
    }

    pub(crate) fn into_work(self) -> W {
        self.0
    }

    /// Starts the work now. A transfer of host memory that is not pinned completes before this
    /// returns, since the device must not use memory the runtime cannot keep alive. A transfer
    /// first waits for the device work submitted before it on its bytes, and dropping it before
    /// that wait ends cancels it.
    pub fn submit(self) -> Submitted<W::Future> {
        self.0.submit()
    }
}

impl<W: Work> IntoFuture for Prepared<W> {
    type Output = Result<W::Output, Error>;
    type IntoFuture = Submitted<W::Future>;

    fn into_future(self) -> Self::IntoFuture {
        self.submit()
    }
}

impl<W> Debug for Prepared<W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Prepared").finish_non_exhaustive()
    }
}

/// Started work: a launch or a host transfer, which runs to completion even when this Future is
/// dropped or forgotten.
#[must_use = "await the work to observe its result"]
pub struct Submitted<F>(F);

impl<F> Debug for Submitted<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Submitted").finish_non_exhaustive()
    }
}

impl<F: Future> Future for Submitted<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        // SAFETY: the started work is pinned exactly when this is.
        unsafe { self.map_unchecked_mut(|this| &mut this.0) }.poll(cx)
    }
}

/// A launch: failed before it ran, running on a device, or run by the host, until its result is
/// taken.
#[doc(hidden)]
pub enum Launched<O> {
    /// The launch failed before it ran.
    Failed(Error),
    /// A device runs it; `O` is ready once the completion reports.
    Running(O, furiosa_opt_rt::Launch),
    /// The host ran it.
    Ran(O),
    /// Its result was taken.
    Taken,
}

impl<O> Debug for Launched<O> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Launched").finish_non_exhaustive()
    }
}

impl Submitted<Launched<()>> {
    /// A launch the backend submitted, or ran already when it holds no completion.
    pub(crate) fn launched(result: Result<Option<furiosa_opt_rt::Launch>, Error>) -> Self {
        Self(match result {
            Ok(Some(completion)) => Launched::Running((), completion),
            Ok(None) => Launched::Ran(()),
            Err(error) => Launched::Failed(error),
        })
    }

    /// This launch, yielding `output` once it completes.
    fn returning<O>(self, output: O) -> Submitted<Launched<O>> {
        Submitted(match self.0 {
            Launched::Failed(error) => Launched::Failed(error),
            Launched::Running((), completion) => Launched::Running(output, completion),
            Launched::Ran(()) => Launched::Ran(output),
            Launched::Taken => Launched::Taken,
        })
    }
}

impl<O> Future for Launched<O> {
    type Output = Result<O, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: neither the output nor the Unpin completion handle is structurally pinned.
        let this = unsafe { self.get_unchecked_mut() };
        if let Self::Running(_, completion) = this
            && let Err(error) = ready!(Pin::new(completion).poll(cx))
        {
            *this = Self::Taken;
            // A device task's panic panics the awaiting caller, as the function's Rust body does
            // on the host backend.
            if let DeviceError::Panicked(message) = error {
                panic!("{message}");
            }
            return Poll::Ready(Err(error.into()));
        }
        Poll::Ready(match std::mem::replace(this, Self::Taken) {
            Self::Failed(error) => Err(error),
            Self::Running(output, _) | Self::Ran(output) => Ok(output),
            Self::Taken => panic!("launch polled after completion"),
        })
    }
}

/// A host transfer is work a [`Prepared`] starts by polling it once.
impl<O, F: Future<Output = Result<O, Error>>> Work for F {
    type Output = O;
    type Future = Issued<F>;

    fn submit(self) -> Submitted<Issued<F>> {
        Submitted(Issued::new(self))
    }
}

/// A host transfer issued by one poll at submission, then pending or ready until its result is
/// taken; the awaiting task's poll registers its own waker.
#[doc(hidden)]
pub enum Issued<F: Future> {
    /// The transfer still runs.
    Pending(Pin<Box<F>>),
    /// The transfer finished within its first poll.
    Ready(F::Output),
    /// Its result was taken.
    Taken,
}

impl<F: Future> Debug for Issued<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Issued").finish_non_exhaustive()
    }
}

impl<F: Future> Issued<F> {
    fn new(transfer: F) -> Self {
        let mut transfer = Box::pin(transfer);
        // rt finishes a transfer of host memory it does not own within this poll, so a forgotten
        // `Submitted` leaves the device no borrowed bytes (rt `Device::write` / `read`).
        match transfer.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(output) => Self::Ready(output),
            Poll::Pending => Self::Pending(transfer),
        }
    }
}

impl<F: Future> Future for Issued<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        // SAFETY: the transfer is pinned in its own box, and the finished output is not pinned.
        let this = unsafe { self.get_unchecked_mut() };
        match std::mem::replace(this, Self::Taken) {
            Self::Pending(mut transfer) => {
                let poll = transfer.as_mut().poll(cx);
                if poll.is_pending() {
                    *this = Self::Pending(transfer);
                }
                poll
            }
            Self::Ready(output) => Poll::Ready(output),
            Self::Taken => panic!("transfer polled after completion"),
        }
    }
}

/// Prepares a launch of `function` with `args`; `.await` submits it and waits, `.submit()` only
/// submits, and `.output(&mut out)` writes its returns into existing storage. The first launch of
/// a function on a device loads it on the calling thread; `function.load(&device)` does that up
/// front.
pub fn launch<F, Args>(function: F, args: Args) -> Prepared<Launch<F, Args>>
where
    F: DeviceFn<Args>,
    Args: DeviceSend,
{
    Prepared(Launch {
        function,
        args,
        output: Returned,
    })
}

/// A launch, as a [`Prepared`] holds it: into storage it returns, or into `output` once
/// [`Prepared::output`] names it.
#[doc(hidden)]
#[derive(Debug)]
pub struct Launch<F, Args, Output = Returned> {
    function: F,
    args: Args,
    output: Output,
}

/// The returns of a launch go into storage the launch returns.
#[doc(hidden)]
#[derive(Debug)]
pub struct Returned;

impl<F: DeviceFn<Args>, Args: DeviceSend, R: Returns<F::Output>> Work for Launch<F, Args, R> {
    type Output = R::Value;
    type Future = tracing::instrument::Instrumented<Launched<R::Value>>;

    fn submit(self) -> Submitted<Self::Future> {
        // No `type_name`: for a function with ADT const arguments it makes rustc abort a
        // compilation that emits no diagnostic, and the NPU spans inside already name the kernel.
        let span = tracing::info_span!("launch", cat = "launch");
        let launched = span.in_scope(|| self.function.launch(self.args, self.output));
        Submitted(launched.0.instrument(span))
    }
}

impl<O> Submitted<tracing::instrument::Instrumented<Launched<O>>> {
    /// This launch once the backend accepted it, or the error that refused it before it ran. An
    /// accepted launch that fails reports its error only when awaited.
    pub fn accepted(mut self) -> Result<Self, Error> {
        match std::mem::replace(self.0.inner_mut(), Launched::Taken) {
            Launched::Failed(error) => Err(error),
            launched => {
                *self.0.inner_mut() = launched;
                Ok(self)
            }
        }
    }
}

impl<F: DeviceFn<Args>, Args: DeviceSend> Prepared<Launch<F, Args>> {
    /// Writes the function's returns into `output` instead of new storage.
    pub fn output(self, output: &mut F::Output) -> Prepared<Launch<F, Args, &mut F::Output>> {
        let Launch { function, args, .. } = self.0;
        Prepared(Launch { function, args, output })
    }
}

/// A device function's parameters after its leading `&mut Device`, as [`crate::Device::launch`] takes
/// them; `prepend` puts the context back in front, in the shape the function's [`DeviceFn`] names.
pub trait Prepend<Head> {
    /// The parameters with `Head` in front: a lone `Head` when there are none.
    type Output;
    /// `head`, then `self`'s elements.
    fn prepend(self, head: Head) -> Self::Output;
}

impl<Head> Prepend<Head> for () {
    type Output = Head;

    fn prepend(self, head: Head) -> Head {
        head
    }
}

macro_rules! impl_prepend {
    (@expand [$($T:ident,)*]) => {};
    (@expand [$($T:ident,)*] $head:ident $(, $tail:ident)*) => {
        impl_prepend!($($T,)* $head);
        impl_prepend!(@expand [$($T,)* $head,] $($tail),*);
    };
    ($($T:ident),+) => {
        #[expect(non_snake_case, reason = "type parameters are destructuring names")]
        impl<Head, $($T,)+> Prepend<Head> for ($($T,)+) {
            type Output = (Head, $($T,)+);

            fn prepend(self, head: Head) -> Self::Output {
                let ($($T,)+) = self;
                (head, $($T,)+)
            }
        }
    };
}

impl_prepend!(@expand [] A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);
