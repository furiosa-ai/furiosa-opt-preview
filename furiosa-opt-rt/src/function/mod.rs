//! Eager execution with device-owned operands and the compiler's epilogues.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use furiosa_opt_ipc::{Profile, Records};

use crate::memory::Kind;
use crate::{Access, Buffer, BufferSlice, Device, Error, Result};

mod argument;
mod completion;
mod slots;
mod table;
#[cfg(test)]
mod tests;
pub(crate) use argument::Operand;
pub use argument::{Argument, Arguments};
use completion::Completion;
pub(crate) use completion::Launches;
use slots::Slots;
#[cfg(test)]
pub(crate) use table::Inline;
pub(crate) use table::Prepared;
use table::Table;

/// A loaded program that launches on its device with arguments its binder turns into operands.
pub struct Function {
    launch: Box<Callable>,
    ordered: bool,
}

impl Function {
    /// Stages `bytes` on every chip and allocates slots for `capacity` arguments once.
    /// Launch temporaries hold SRAM below `sram_end`. `bind` turns a launch's arguments
    /// into operands and returns its [`Epilogue`], which runs only after the hardware finished;
    /// polling an epilogue must not block or launch a function, and its readback must use pinned
    /// host views.
    ///
    /// # Safety
    /// Binding must declare every caller-memory access with its mode; memory reached through an
    /// operand's contents must be declared through `launch_holding`.
    /// Device code must stay within those regions; writes to compiler storage need their own ordering proof.
    pub async unsafe fn load<F: Epilogue>(
        device: &Arc<Device>,
        bytes: &[u8],
        sram_end: usize,
        capacity: usize,
        bind: impl for<'a> Fn(&mut Supplied<'_, 'a>, &mut Arguments) -> Result<F> + Send + Sync + 'static,
    ) -> Result<Self> {
        let mut staging = device.alloc(Kind::Dram, bytes.len())?;
        device
            .write(std::iter::repeat(bytes.into()).zip(staging.slice_mut(..)?.chips()))
            .await?;
        let executable = Executable {
            device: Arc::clone(device),
            token: staging.generation(),
            staging: Arc::new(staging),
            sram_end,
        };
        u32::try_from(capacity).map_err(|_| Error::Invalid("argument count exceeds u32".into()))?;
        let device = &executable.device;
        let slots = Slots::<F>::new(device, capacity)?;
        device.launches.push(slots.clone());
        Ok(Self {
            launch: Box::new(move |args, held, ordered| {
                slots.launch(&executable, held, |arguments| {
                    arguments.set_owner(if ordered {
                        executable.device.allocator.owner()?
                    } else {
                        crate::buffer::Owner::Unordered
                    });
                    bind(args, arguments)
                })
            }),
            ordered: false,
        })
    }

    /// Allows overlapping caller memory between distinct ordered launches on this device.
    ///
    /// # Safety
    /// Every chip must finish caller-memory accesses before its first cluster answers, including
    /// indirect and host DMA. Peer-shared memory stays exclusive; generic tasks must not opt in.
    pub unsafe fn ordered(self) -> Self {
        Self { ordered: true, ..self }
    }

    /// Publishes before returning; dropping the observer does not cancel execution.
    pub fn launch<'a>(&self, args: impl Supply<'a>) -> Result<Launch> {
        self.launch_holding(args, &[])
    }

    /// Also holds memory reached through operand contents until hardware completion.
    /// Each region must belong to this device and cover every chip; dropping the observer releases nothing.
    pub fn launch_holding<'a>(&self, args: impl Supply<'a>, held: &[(BufferSlice<'a>, Access)]) -> Result<Launch> {
        let mut args = Some(args);
        let mut supplied = |bind: &mut dyn FnMut(Argument<'a>) -> Result<()>| {
            args.take().expect("a launch binds its arguments once").supply(bind)
        };
        (self.launch)(&mut supplied, held, self.ordered)
    }
}

/// What a launch takes its arguments from, in operand order: any iterator of them, or [`Each`],
/// whose caller hands them out one at a time and so holds no list of them.
pub trait Supply<'a> {
    /// Hands every argument to `bind` in order, stopping at the first it refuses.
    fn supply(self, bind: &mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()>;
}

impl<'a, I: IntoIterator<Item = Argument<'a>>> Supply<'a> for I {
    fn supply(self, bind: &mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()> {
        self.into_iter().try_for_each(bind)
    }
}

/// Arguments `F` hands to its callback one at a time, in operand order.
pub struct Each<F>(F);

impl<F> Each<F> {
    pub fn new<'a>(supply: F) -> Self
    where
        F: FnOnce(&mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()>,
    {
        Self(supply)
    }
}

impl<'a, F: FnOnce(&mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()>> Supply<'a> for Each<F> {
    fn supply(self, bind: &mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()> {
        (self.0)(bind)
    }
}

/// One launch's arguments for a binder: called once, it hands each to its callback in operand order.
pub type Supplied<'s, 'a> = dyn FnMut(&mut dyn FnMut(Argument<'a>) -> Result<()>) -> Result<()> + 's;

/// One submitted launch of a [`Function`], resolving to the outcome of its hardware and epilogue.
/// Dropping it cancels nothing.
#[must_use = "await the launch to observe its result"]
pub struct Launch {
    completion: Arc<dyn Completion>,
    index: usize,
    generation: u64,
}

impl Future for Launch {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let started = Instant::now();
        loop {
            let result = self.completion.poll(self.index, self.generation, context);
            if result.is_ready() || started.elapsed() >= SPIN {
                return result;
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for Launch {
    fn drop(&mut self) {
        self.completion.release(self.index, self.generation);
    }
}

/// What one launch runs on every cluster's reply once its hardware finished, and the profile it
/// asks every chunk to record into those replies. A future is an epilogue that asks for none.
pub trait Epilogue: Send + 'static {
    /// What every chunk records into the replies, if anything.
    fn profile(&self) -> Option<Profile>;

    /// Runs on each cluster's records, in cluster order, until it is ready.
    fn run(self: Pin<&mut Self>, replies: &[Records]) -> Poll<Result<()>>;
}

impl<F: Future<Output = Result<()>> + Send + 'static> Epilogue for F {
    fn profile(&self) -> Option<Profile> {
        None
    }

    fn run(self: Pin<&mut Self>, _: &[Records]) -> Poll<Result<()>> {
        self.poll(&mut Context::from_waker(Waker::noop()))
    }
}

/// A function's compiled program in device DRAM.
pub(crate) struct Executable {
    pub(crate) device: Arc<Device>,
    pub(crate) staging: Arc<Buffer>,
    /// The staging allocation's generation, which tells this image from another staged later at
    /// the same address.
    pub(crate) token: u64,
    pub(crate) sram_end: usize,
}

type Callable =
    dyn for<'a> Fn(&mut Supplied<'_, 'a>, &[(BufferSlice<'a>, Access)], bool) -> Result<Launch> + Send + Sync;

/// How long a poll waits for a launch to finish before it returns `Pending`: a launch finishing
/// sooner is seen without a wake-up round trip through the executor.
const SPIN: Duration = Duration::from_micros(10);
