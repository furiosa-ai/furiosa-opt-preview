//! Eager execution with device-owned operands and the compiler's epilogues.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use furiosa_opt_ipc::{Profile, Records};

use crate::buffer::Waiters;
use crate::device::Contention;
use crate::driver::mailbox::IN_FLIGHT;
use crate::driver::{Driver, Replies, Ticket};
use crate::memory::Kind;
use crate::{Access, Buffer, BufferSlice, Device, Error, Result};

mod argument;
mod table;
pub(crate) use argument::Operand;
pub use argument::{Argument, Arguments};
use table::Table;
pub(crate) use table::{Inline, Prepared};

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
        bind: impl for<'a> Fn(&mut dyn Iterator<Item = Argument<'a>>, &mut Arguments) -> Result<F> + Send + Sync + 'static,
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
        let slots = Arc::new(Slots {
            slots: (0..IN_FLIGHT)
                .map(|_| {
                    Ok(Slot {
                        launched: AtomicU64::new(0),
                        ended: AtomicU64::new(0),
                        completed: AtomicU64::new(0),
                        released: AtomicU64::new(0),
                        running: AtomicBool::new(false),
                        state: Mutex::new(State {
                            hardware: None,
                            future: None,
                            arguments: Arguments::new(capacity, Arc::downgrade(device)),
                            indirect: Table::new(device, capacity)?,
                            generation: 0,
                            result: Ok(()),
                            waker: None,
                        }),
                    })
                })
                .collect::<Result<_>>()?,
            failure: OnceLock::new(),
            worker: device.worker.waker().clone(),
            waiters: Mutex::default(),
        });
        device.launches.push(slots.clone());
        Ok(Self {
            launch: Box::new(move |args, held, ordered| {
                slots.launch(&executable, held, |arguments| {
                    arguments.set_owner(if ordered {
                        executable.device.allocations.owner()?
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
    pub fn launch<'a>(&self, args: impl IntoIterator<Item = Argument<'a>>) -> Result<Launch> {
        self.launch_holding(args, &[])
    }

    /// Also holds memory reached through operand contents until hardware completion.
    /// Each region must belong to this device and cover every chip; dropping the observer releases nothing.
    pub fn launch_holding<'a>(
        &self,
        args: impl IntoIterator<Item = Argument<'a>>,
        held: &[(BufferSlice<'a>, Access)],
    ) -> Result<Launch> {
        (self.launch)(&mut args.into_iter(), held, self.ordered)
    }
}

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

/// Every function's launch slots on one device, polled by its worker so that launches nobody
/// observes still complete and run their epilogues.
#[derive(Default)]
pub(crate) struct Launches(Mutex<Vec<Arc<dyn Completion>>>);

/// Neither walk holds the list while it polls, since an epilogue's transfer may complete launches.
impl Launches {
    /// Collects every function's finished launches; true when none is in flight. Skipping
    /// contended slots is the worker's walk, which alone removes a function nothing holds any
    /// more, so the one it removes is the one at `index`. Waiting walks down, so a concurrent
    /// removal shifts only functions it already passed.
    pub(crate) fn complete(&self, contention: Contention) -> bool {
        let mut idle = true;
        match contention {
            Contention::Skip => {
                let mut index = 0;
                while let Some(function) = self.get(index) {
                    let complete = function.complete(contention);
                    idle &= complete;
                    let mut functions = self.0.lock().unwrap();
                    // The list and this clone: no Function or Launch holds the slots any more.
                    if complete && Arc::strong_count(&function) == 2 {
                        functions.remove(index);
                    } else {
                        index += 1;
                    }
                }
            }
            Contention::Wait => {
                let mut index = self.0.lock().unwrap().len();
                while let Some(next) = index.checked_sub(1) {
                    index = next;
                    if let Some(function) = self.get(index) {
                        idle &= function.complete(contention);
                    }
                }
            }
        }
        idle
    }

    /// The launches submitted so far that have not ended.
    pub(crate) fn launched(&self) -> Launched {
        Launched(
            (self.0.lock().unwrap().iter())
                .filter_map(|function| Some((Arc::clone(function), function.launched()?)))
                .collect(),
        )
    }

    fn push(&self, function: Arc<dyn Completion>) {
        self.0.lock().unwrap().push(function);
    }

    fn get(&self, index: usize) -> Option<Arc<dyn Completion>> {
        self.0.lock().unwrap().get(index).cloned()
    }
}

/// Launches of a device, each by its function and the generation of each of its slots.
pub(crate) struct Launched(Vec<(Arc<dyn Completion>, [u64; IN_FLIGHT])>);

impl Launched {
    /// Whether every launch ended; `waker` is woken at the next end otherwise.
    pub(crate) fn ended(&self, waker: &Waker) -> bool {
        self.0
            .iter()
            .all(|(function, launched)| function.ended(launched, waker))
    }
}

type Callable = dyn for<'a> Fn(&mut dyn Iterator<Item = Argument<'a>>, &[(BufferSlice<'a>, Access)], bool) -> Result<Launch>
    + Send
    + Sync;

trait Completion: Send + Sync {
    /// Collects the launches the hardware finished and runs their epilogues; true when none of
    /// them is in flight.
    fn complete(&self, contention: Contention) -> bool;
    fn poll(&self, index: usize, generation: u64, context: &mut Context<'_>) -> Poll<Result<()>>;
    /// The observer of launch `generation` in slot `index` is gone.
    fn release(&self, index: usize, generation: u64);
    /// The generation each slot launched last, unless every one of them ended.
    fn launched(&self) -> Option<[u64; IN_FLIGHT]>;
    /// Whether every launch of `launched` ended; `waker` is woken at the next end otherwise.
    fn ended(&self, launched: &[u64; IN_FLIGHT], waker: &Waker) -> bool;
}

struct Slots<F> {
    slots: Box<[Slot<F>]>,
    failure: OnceLock<Error>,
    worker: Waker,
    /// Woken whenever one of these launches ends.
    waiters: Mutex<Waiters>,
}

impl<F: Epilogue> Slots<F> {
    fn launch(
        self: &Arc<Self>,
        executable: &Executable,
        held: &[(BufferSlice<'_>, Access)],
        bind: impl FnOnce(&mut Arguments) -> Result<F>,
    ) -> Result<Launch> {
        let (index, mut slot) = self.wait(|| {
            Ok(self.slots.iter().enumerate().find_map(|(index, slot)| {
                let state = try_lock(&slot.state)?;
                // A failed launch keeps its slot while its observer may still ask for the error.
                let reusable = state.result.is_ok() || slot.released.load(Ordering::Acquire) >= state.generation;
                (state.hardware.is_none() && state.future.is_none() && reusable).then_some((index, state))
            }))
        })?;
        let generation = slot.generation.checked_add(1).ok_or(Error::Capacity)?;
        let submitted = bind(&mut slot.arguments).and_then(|future| {
            slot.arguments.reserve(held.len());
            slot.hardware = Some(self.submit(executable, &mut slot, held, future.profile())?);
            slot.future = Some((future, Arc::clone(&executable.device.driver), Replies::new()));
            Ok(())
        });
        if let Err(error) = submitted {
            slot.arguments.clear();
            return Err(error);
        }
        self.slots[index].running.store(true, Ordering::Release);
        self.slots[index].launched.store(generation, Ordering::Release);
        slot.generation = generation;
        self.worker.wake_by_ref();
        Ok(Launch {
            completion: self.clone(),
            index,
            generation,
        })
    }

    /// Admission waits for a free slot and queue room, which the worker's progress releases.
    fn wait<T>(&self, mut attempt: impl FnMut() -> Result<Option<T>>) -> Result<T> {
        loop {
            if let Some(error) = self.failure.get() {
                return Err(error.clone());
            }
            if let Some(value) = attempt()? {
                return Ok(value);
            }
            self.worker.wake_by_ref();
            std::thread::yield_now();
        }
    }

    fn submit(
        &self,
        executable: &Executable,
        state: &mut State<F>,
        held: &[(BufferSlice<'_>, Access)],
        profile: Option<Profile>,
    ) -> Result<Ticket> {
        let device = &executable.device;
        let chips = u32::from(device.topology().chips);
        for operand in state.arguments.operands() {
            match operand {
                Operand::Unbound | Operand::Value(_) => {}
                Operand::Device {
                    buffer,
                    chips: selected,
                    ..
                } => {
                    if !buffer.belongs_to(&device.allocations) {
                        return Err(Error::ForeignBuffer);
                    }
                    if selected.count_ones() != chips {
                        return Err(Error::Invalid(
                            "a launch operand must cover every chip of the device".into(),
                        ));
                    }
                }
                Operand::Host { registration, .. } => {
                    if !registration.buffer.belongs_to(&device.allocations) {
                        return Err(Error::ForeignBuffer);
                    }
                }
            }
        }
        for (view, _) in held {
            if !view.buffer().belongs_to(&device.allocations) {
                return Err(Error::ForeignBuffer);
            }
            if view.chip_mask().count_ones() != chips {
                return Err(Error::Invalid("held memory must cover every chip of the device".into()));
            }
        }
        // The clusters read the staged image while the launch runs.
        let image = executable.staging.slice(..).expect("a buffer holds its own range");
        device.admit(|| {
            state
                .arguments
                .acquire(held.iter().copied().chain([(image, Access::Read)]))
        })?;
        let prepared = match Inline::new(device, state.arguments.operands()) {
            Some(inline) => Prepared::Inline(inline),
            None => {
                let table = state.indirect.as_mut().expect("capacity allocates an indirect table");
                let bytes = table.encode(device, state.arguments.operands());
                if !table.cached(bytes) {
                    let mut empty = None;
                    for slot in &self.slots {
                        let Some(mut idle) = try_lock(&slot.state) else {
                            continue;
                        };
                        if idle.hardware.is_none() && idle.future.is_none() {
                            let other = idle.indirect.as_mut().expect("slots share argument capacity");
                            if table.reuse(other, bytes) {
                                empty = None;
                                break;
                            }
                            if empty.is_none() && other.empty() {
                                empty = Some(idle);
                            }
                        }
                    }
                    if let Some(mut idle) = empty {
                        table.swap(idle.indirect.as_mut().expect("slots share argument capacity"));
                    }
                }
                Prepared::Indirect(table.upload(device, state.arguments.operands().len())?)
            }
        };
        self.wait(|| device.driver.submit(executable, &prepared, profile))
    }

    fn advance(&self, slot: &Slot<F>) -> bool {
        try_lock(&slot.state).is_some_and(|state| self.step(slot, state))
    }

    /// Collects `slot`'s finished hardware, then runs its epilogue; true once the slot is idle.
    fn step(&self, slot: &Slot<F>, mut state: MutexGuard<'_, State<F>>) -> bool {
        let mut ended = false;
        if state.future.is_some()
            && let Poll::Ready(result) = state.poll(&slot.running)
        {
            state.future = None;
            match &result {
                // Failed hardware fails every later launch; a failed epilogue only its own.
                Err(error) if state.hardware.is_some() => {
                    self.failure.get_or_init(|| error.clone());
                }
                Err(_) => {}
                // Successful receipts remain observable while a later call holds this slot.
                Ok(()) => slot.completed.store(state.generation, Ordering::Release),
            }
            slot.ended.store(state.generation, Ordering::Release);
            state.result = result;
            ended = true;
        }
        // Failed hardware operands stay in their slot until the device stops.
        let idle = state.hardware.is_none() && state.future.is_none();
        let waker = state.future.is_none().then(|| state.waker.take()).flatten();
        drop(state);
        if let Some(waker) = waker {
            waker.wake();
        }
        if ended {
            let waiters = self.waiters.lock().unwrap().take();
            waiters.wake();
        }
        idle
    }
}

impl<F: Epilogue> Completion for Slots<F> {
    /// Waiting advances every running slot once, spinning while another thread holds it; an
    /// admission holds a slot that is not running, so this never waits for one.
    fn complete(&self, contention: Contention) -> bool {
        if contention == Contention::Wait {
            for slot in &self.slots {
                while slot.running.load(Ordering::Acquire) {
                    if let Some(state) = try_lock(&slot.state) {
                        self.step(slot, state);
                        break;
                    }
                    std::thread::yield_now();
                }
            }
        }
        self.slots.iter().fold(true, |idle, slot| self.advance(slot) & idle)
    }

    fn release(&self, index: usize, generation: u64) {
        self.slots[index].released.fetch_max(generation, Ordering::Release);
    }

    fn launched(&self) -> Option<[u64; IN_FLIGHT]> {
        let launched = std::array::from_fn(|index| self.slots[index].launched.load(Ordering::Acquire));
        let ended = |(slot, generation): (&Slot<F>, &u64)| slot.ended.load(Ordering::Acquire) >= *generation;
        (!self.slots.iter().zip(&launched).all(ended)).then_some(launched)
    }

    fn ended(&self, launched: &[u64; IN_FLIGHT], waker: &Waker) -> bool {
        let ended = || {
            (self.slots.iter().zip(launched))
                .all(|(slot, generation)| slot.ended.load(Ordering::Acquire) >= *generation)
        };
        ended() || {
            self.waiters.lock().unwrap().register(waker);
            ended()
        }
    }

    fn poll(&self, index: usize, generation: u64, context: &mut Context<'_>) -> Poll<Result<()>> {
        let slot = &self.slots[index];
        if generation <= slot.completed.load(Ordering::Acquire) {
            return Poll::Ready(Ok(()));
        }
        self.advance(slot);
        let Some(mut state) = try_lock(&slot.state) else {
            context.waker().wake_by_ref();
            return Poll::Pending;
        };
        if generation != state.generation {
            return Poll::Ready(Ok(()));
        }
        if state.future.is_none() {
            return Poll::Ready(state.result.clone());
        }
        state
            .waker
            .get_or_insert_with(|| context.waker().clone())
            .clone_from(context.waker());
        Poll::Pending
    }
}

struct Slot<F> {
    /// The latest generation submitted.
    launched: AtomicU64,
    /// The latest generation whose hardware and epilogue ended, either way.
    ended: AtomicU64,
    completed: AtomicU64,
    /// The latest generation whose observer is gone.
    released: AtomicU64,
    /// The hardware holds the operands: set at submission, cleared once they are released.
    running: AtomicBool,
    state: Mutex<State<F>>,
}

struct State<F> {
    /// The launch on the clusters, until every one answered.
    hardware: Option<Ticket>,
    /// The epilogue, the driver it waits on and, once every cluster answered, their replies it
    /// runs on: the device itself goes with its last handle.
    future: Option<(F, Arc<Driver>, Replies)>,
    arguments: Arguments,
    indirect: Option<Table>,
    generation: u64,
    result: Result<()>,
    waker: Option<Waker>,
}

impl<F: Epilogue> State<F> {
    /// The worker polls every active slot while anything runs, so epilogues need no waker.
    /// `running` falls once the hardware released the operands, before the epilogue runs.
    fn poll(&mut self, running: &AtomicBool) -> Poll<Result<()>> {
        let (future, driver, replies) = self.future.as_mut().expect("active launch");
        if let Some(ticket) = &mut self.hardware {
            match driver.poll(ticket) {
                Ok(None) => return Poll::Pending,
                Ok(Some(answered)) => {
                    *replies = answered;
                    self.hardware = None;
                    self.arguments.clear();
                    running.store(false, Ordering::Release);
                }
                Err(error) => {
                    self.arguments.poison();
                    return Poll::Ready(Err(error));
                }
            }
        }
        // SAFETY: the future stays in this fixed slot until it is dropped in place.
        unsafe { Pin::new_unchecked(future) }.run(replies)
    }
}

impl<F> Drop for State<F> {
    fn drop(&mut self) {
        if self.hardware.is_some() {
            self.arguments.poison();
        }
    }
}

// Binding one call may wait for resources released by another call's epilogue.
// Progress must skip that slot rather than wait for its bind callback.
fn try_lock<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::WouldBlock) => None,
        Err(TryLockError::Poisoned(error)) => panic!("{error}"),
    }
}

/// How long a poll waits for a launch to finish before it returns `Pending`: a launch finishing
/// sooner is seen without a wake-up round trip through the executor.
const SPIN: Duration = Duration::from_micros(10);

#[cfg(test)]
mod tests {
    use std::marker::PhantomPinned;

    use super::*;
    use crate::test::{Wakes, create, function, poll, woken};
    use arrayvec::ArrayVec;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn observe(launch: &mut Launch) -> Poll<Result<()>> {
        Pin::new(launch).poll(&mut Context::from_waker(Waker::noop()))
    }

    /// A launch, its completion and its slot's reuse allocate nothing, whether its arguments fit
    /// the frame or go in a table that changes with every launch.
    #[test]
    fn submits_without_allocations() {
        for count in [1, furiosa_opt_ipc::MAX_LAUNCH_ARGS + 1] {
            let device = Arc::new(crate::test::with(&[0], 1, Vec::new(), 1 << 16));
            let function = crate::test::create(&device, count, |args, bound| {
                args.enumerate()
                    .try_for_each(|(index, argument)| bound.bind(index, argument))?;
                Ok(std::future::ready(Ok(())))
            });
            let input = device.alloc(Kind::Dram, 1).unwrap();
            let mut launches = ArrayVec::<Launch, IN_FLIGHT>::new();
            let round = |launches: &mut ArrayVec<Launch, IN_FLIGHT>, round: u64| {
                for index in 0..launches.capacity() as u64 {
                    let values = (1..count as u64).map(|value| Argument::Value(value + (round * 100 + index) * 10000));
                    let args = std::iter::once(input.slice(..).unwrap().into()).chain(values);
                    launches.push(function.launch(args).unwrap());
                }
                device.launches.complete(Contention::Skip);
                for launch in launches.iter_mut() {
                    assert_eq!(observe(launch), Poll::Ready(Ok(())), "{count} arguments");
                }
                launches.clear();
            };
            // The first round sizes the runtime's pools.
            round(&mut launches, 0);
            crate::test::without_allocations(|| (1..3).for_each(|index| round(&mut launches, index)));
        }
    }

    fn ordered(device: &Arc<Device>) -> Function {
        // SAFETY: the fixture never executes code; it tests the compiled-kernel admission contract.
        let load = unsafe {
            Function::load(
                device,
                &[0],
                crate::arch::rngd::sram::WINDOW.start as usize,
                8,
                |args, bound| {
                    args.enumerate()
                        .try_for_each(|(index, argument)| bound.bind(index, argument))?;
                    Ok(std::future::ready(Ok(())))
                },
            )
        };
        match std::pin::pin!(load).poll(&mut Context::from_waker(Waker::noop())) {
            // SAFETY: the fixture has no caller-memory accesses on any cluster.
            Poll::Ready(function) => unsafe { function.unwrap().ordered() },
            Poll::Pending => panic!("the fixture's small BAR4 write must finish synchronously"),
        }
    }

    /// Distinct ordered launches of one device share memory before completion, aliases and indirect
    /// uses included, except peer-visible memory; every other overlap is excluded.
    #[test]
    fn admits_overlap_only_between_ordered_launches() {
        for (chips, kind) in [
            (&[0][..], Kind::Dram),
            (&[0][..], Kind::Sram),
            (&[0][..], Kind::DramShared),
            (&[0, 1][..], Kind::Dram),
            (&[0, 1][..], Kind::Sram),
        ] {
            for access in [Access::Read, Access::Write] {
                for case in 0..4 {
                    let (device, file) = crate::test::running(chips, 8);
                    let first = ordered(&device);
                    let other = ordered(&device);
                    let buffer = device.alloc(kind, 256).unwrap();
                    let alias = buffer.alias(..).unwrap();
                    let mut before = first.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap();
                    let function = if case == 1 { &other } else { &first };
                    let view = if case == 2 { alias.slice(..) } else { buffer.slice(..) }.unwrap();
                    let after = match case {
                        3 => function.launch_holding([], &[(view, access)]),
                        _ => function.launch([match access {
                            Access::Read => Argument::Read(view),
                            Access::Write => Argument::Write(view),
                        }]),
                    };
                    let mut after = after.unwrap_or_else(|error| panic!("{kind:?}/{access:?}/case {case}: {error:?}"));
                    assert_eq!(observe(&mut before), Poll::Pending);
                    assert_eq!(observe(&mut after), Poll::Pending);
                    crate::test::answer(&file, 2);
                    assert!(device.launches.complete(Contention::Skip));
                }
            }
        }

        // Distinct ordered launches, including different functions, may share device writes.
        {
            // Given: file-backed hardware does not access caller memory.
            let (device, file) = crate::test::running(&[0], 8);
            // SAFETY: the stub has no memory accesses on any cluster.
            let first = unsafe { function(&device).ordered() };
            // SAFETY: the stub has no memory accesses on any cluster.
            let second = unsafe { function(&device).ordered() };
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            drop(first.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
            // When: another ordered launch names the same bytes through indirect holdings.
            let admitted = second.launch_holding([], &[(buffer.slice(..).unwrap(), Access::Write)]);
            // Then: admission succeeds before either completion is observed.
            assert!(
                admitted.is_ok(),
                "ordered HBM overlap must be admitted: {:?}",
                admitted.err()
            );
            crate::test::answer(&file, 2);
            device.launches.complete(Contention::Skip);
        }

        // Ordered host leases share one registration while the earlier launch remains in flight.
        {
            let (device, file) = crate::test::running(&[0], 8);
            // SAFETY: the stub has no memory accesses on any cluster.
            let function = unsafe { function(&device).ordered() };
            let mut host = crate::test::mapped(&device, vec![0], 4096);
            drop(function.launch([Argument::Host(host.slice_mut(..))]).unwrap());
            let admitted = function.launch([Argument::Host(host.slice_mut(..))]);
            assert!(
                admitted.is_ok(),
                "ordered Host overlap must be admitted: {:?}",
                admitted.err()
            );
            crate::test::answer(&file, 2);
            device.launches.complete(Contention::Skip);
        }

        // Peer-visible memory excludes conflicting compiled launches, including indirect uses.
        {
            for access in [Access::Read, Access::Write] {
                for case in 0..4 {
                    let (device, file) = crate::test::running(&[0, 1], 8);
                    let function = ordered(&device);
                    let buffer = device.alloc(Kind::DramShared, 256).unwrap();
                    let alias = buffer.alias(..).unwrap();
                    let mut first = match case {
                        0 | 1 => function.launch([Argument::Write(buffer.slice(..).unwrap())]),
                        _ => function.launch_holding([], &[(buffer.slice(..).unwrap(), Access::Write)]),
                    }
                    .unwrap();
                    let view = alias.slice(..).unwrap();
                    let second = match case {
                        0 | 2 => function.launch([match access {
                            Access::Read => Argument::Read(view),
                            Access::Write => Argument::Write(view),
                        }]),
                        _ => function.launch_holding([], &[(view, access)]),
                    };
                    assert!(matches!(second, Err(Error::Busy)), "{access:?}/case {case}");
                    crate::test::answer(&file, 1);
                    assert_eq!(observe(&mut first), Poll::Ready(Ok(())));
                }
            }
        }

        // Generic tasks exclude both generic and compiled dependencies in either submission order.
        {
            for generic_first in [false, true] {
                let (device, file) = crate::test::running(&[0], 8);
                let generic = function(&device);
                let compiled = ordered(&device);
                let buffer = device.alloc(Kind::Dram, 256).unwrap();
                let (first, second) = if generic_first {
                    (&generic, &compiled)
                } else {
                    (&compiled, &generic)
                };
                let mut launch = first.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap();
                assert!(matches!(
                    second.launch([Argument::Read(buffer.slice(..).unwrap())]),
                    Err(Error::Busy)
                ));
                if generic_first {
                    assert!(matches!(
                        generic.launch([Argument::Write(buffer.slice(..).unwrap())]),
                        Err(Error::Busy)
                    ));
                }
                crate::test::answer(&file, 1);
                assert_eq!(observe(&mut launch), Poll::Ready(Ok(())));
                assert!(second.launch([Argument::Write(buffer.slice(..).unwrap())]).is_ok());
                crate::test::answer(&file, 2);
                assert!(device.launches.complete(Contention::Skip));
            }
        }

        // A remapped host allocation cannot share ordered leases across publication domains.
        {
            let (device, file) = crate::test::running(&[0], 8);
            let (other, _) = crate::test::running(&[1], 8);
            // SAFETY: neither stub accesses caller memory.
            let first = unsafe { function(&device).ordered() };
            // SAFETY: neither stub accesses caller memory.
            let second = unsafe { function(&other).ordered() };
            let mut host = crate::test::mapped(&device, vec![0], 4096);
            drop(first.launch([Argument::Host(host.slice_mut(..))]).unwrap());
            host.set_registration(crate::test::mapped(&other, vec![0], 4096).registration().cloned());
            assert_eq!(
                second.launch([Argument::Host(host.slice_mut(..))]).err(),
                Some(Error::Busy)
            );
            crate::test::answer(&file, 1);
            device.launches.complete(Contention::Skip);
            assert!(second.launch([Argument::Host(host.slice_mut(..))]).is_ok());
            other.driver.stop();
        }

        // Only two ordered launches are exempt, in either submission order and memory kind.
        {
            for ordered_first in [false, true] {
                let (device, file) = crate::test::running(&[0], 8);
                // SAFETY: the stub has no memory accesses on any cluster.
                let ordered = unsafe { function(&device).ordered() };
                let unordered = function(&device);
                let (first, second) = if ordered_first {
                    (&ordered, &unordered)
                } else {
                    (&unordered, &ordered)
                };
                let buffer = device.alloc(Kind::Dram, 256).unwrap();
                let mut host = crate::test::mapped(&device, vec![0], 4096);
                drop(
                    first
                        .launch([
                            Argument::Write(buffer.slice(..).unwrap()),
                            Argument::Host(host.slice_mut(..)),
                        ])
                        .unwrap(),
                );
                assert_eq!(
                    second.launch([Argument::Write(buffer.slice(..).unwrap())]).err(),
                    Some(Error::Busy)
                );
                assert_eq!(
                    second.launch([Argument::Host(host.slice_mut(..))]).err(),
                    Some(Error::Busy)
                );
                crate::test::answer(&file, 1);
                device.launches.complete(Contention::Skip);
            }
        }

        // A transfer waits for a launch's HBM operands and is refused its mapped host operands; a
        // launch is refused memory a transfer holds.
        {
            for host_operand in [false, true] {
                for launch_first in [false, true] {
                    let (device, file) = crate::test::running(&[0], 8);
                    // SAFETY: the stub has no memory accesses on any cluster.
                    let function = unsafe { function(&device).ordered() };
                    let buffer = device.alloc(Kind::Dram, 256).unwrap();
                    let mut host = crate::test::mapped(&device, vec![0], 4096);
                    if launch_first {
                        let argument = if host_operand {
                            Argument::Host(host.slice_mut(..))
                        } else {
                            Argument::Write(buffer.slice(..).unwrap())
                        };
                        drop(function.launch([argument]).unwrap());
                        let mut transfer = Box::pin(device.read([(buffer.slice(..).unwrap(), host.slice_mut(..256))]));
                        assert_eq!(
                            transfer.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                            if host_operand {
                                Poll::Ready(Err(Error::Busy))
                            } else {
                                Poll::Pending
                            }
                        );
                        drop(transfer);
                        crate::test::answer(&file, 1);
                        device.launches.complete(Contention::Skip);
                    } else {
                        let mut transfer = Box::pin(device.read([(buffer.slice(..).unwrap(), host.slice_mut(..256))]));
                        assert!(
                            transfer
                                .as_mut()
                                .poll(&mut Context::from_waker(Waker::noop()))
                                .is_pending()
                        );
                        drop(transfer);
                        let argument = if host_operand {
                            Argument::Host(host.slice_mut(..))
                        } else {
                            Argument::Write(buffer.slice(..).unwrap())
                        };
                        assert_eq!(function.launch([argument]).err(), Some(Error::Busy));
                        device.driver.stop();
                    }
                }
            }
        }

        // User leases and aliases within one ordered launch retain their existing exclusion.
        {
            let (device, file) = crate::test::running(&[0], 8);
            // SAFETY: the stub has no memory accesses on any cluster.
            let function = unsafe { function(&device).ordered() };
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            let view = buffer.slice(..).unwrap();
            let lease = crate::Lease::acquire([(view, Access::Read)]).unwrap();
            assert_eq!(function.launch([Argument::Write(view)]).err(), Some(Error::Busy));
            drop(lease);
            assert_eq!(
                function.launch([Argument::Write(view), Argument::Read(view)]).err(),
                Some(Error::Busy)
            );
            assert_eq!(
                function
                    .launch_holding([Argument::Write(view)], &[(view, Access::Read)])
                    .err(),
                Some(Error::Busy)
            );
            let mut host = crate::test::mapped(&device, vec![0], 4096);
            assert_eq!(
                function
                    .launch(host.slice_mut(..512).chunks(256).map(Argument::Host))
                    .err(),
                Some(Error::Busy)
            );
            drop(function.launch([Argument::Write(view)]).unwrap());
            assert_eq!(crate::Lease::acquire([(view, Access::Read)]).err(), Some(Error::Busy));
            crate::test::answer(&file, 1);
            device.launches.complete(Contention::Skip);
        }

        // Non-ordered accesses exclude ordered leases even amid many exempt records.
        {
            use Access::{Read, Write};
            let buffer = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 256).unwrap();
            let whole = buffer.slice(..).unwrap();
            let mut first = crate::Lease::new(1);
            first.set_owner(crate::buffer::Owner::Ordered(1));
            first.extend([(whole, Write)]).unwrap();
            let mut transfer = crate::Lease::new(1);
            assert_eq!(transfer.extend([(whole, Read)]), Err(Error::Busy));
            first.clear();
            transfer.extend([(buffer.slice(128..).unwrap(), Write)]).unwrap();
            first.extend([(buffer.slice(..128).unwrap(), Write)]).unwrap();
            let mut second = crate::Lease::new(1);
            second.set_owner(crate::buffer::Owner::Ordered(2));
            assert_eq!(second.extend([(whole, Write)]), Err(Error::Busy));
            transfer.clear();
            second.extend([(whole, Write)]).unwrap();
            assert_eq!(transfer.extend([(whole, Read)]), Err(Error::Busy));
            first.poison();
            first.clear();
            second.clear();
            assert_eq!(second.extend([(whole, Write)]), Err(Error::Poisoned));
        }
    }

    /// Work holds its memory until the hardware answers, whether it is observed, dropped, forgotten
    /// or cancelled, then releases it.
    #[tokio::test]
    async fn holds_memory_until_the_hardware_answers() {
        let (device, file) = crate::test::running(&[0], 8);
        let function = function(&device);
        let mut buffer = device.alloc(Kind::Dram, 256).unwrap();
        let write = |buffer: &mut Buffer| function.launch([buffer.slice_mut(..).unwrap().into()]).map(drop);
        assert_eq!(write(&mut buffer), Ok(()));
        assert_eq!(write(&mut buffer), Err(Error::Busy));
        crate::test::answer(&file, 1);
        assert_eq!(write(&mut buffer), Ok(()));
        crate::test::answer(&file, 2);
        let bytes = [0; 256];
        let transfer = std::pin::pin!(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
        assert_eq!(
            transfer.poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Ok(()))
        );

        // Each queued launch excludes leases, and delays BAR4 writes, until its own completion,
        // observer or not.
        {
            let (device, file) = crate::test::running(&[0], 8);
            let function = ordered(&device);
            let mut buffer = device.alloc(Kind::Dram, 256).unwrap();
            let mut first = function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap();
            let mut second = Some(function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
            crate::test::answer(&file, 1);
            assert_eq!(observe(&mut first), Poll::Ready(Ok(())));
            let bytes = [0; 256];
            for _ in 0..2 {
                assert!(matches!(
                    crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Read)]),
                    Err(Error::Busy)
                ));
                let write = std::pin::pin!(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
                assert_eq!(write.poll(&mut Context::from_waker(Waker::noop())), Poll::Pending);
                drop(second.take());
            }
            crate::test::answer(&file, 2);
            assert!(device.launches.complete(Contention::Skip));
            assert!(crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]).is_ok());
            let write = std::pin::pin!(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
            assert_eq!(write.poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(Ok(())));
        }

        // A queued dependency keeps its memory until every cluster answers that launch.
        {
            let (device, files) = crate::test::clusters(8, 4096);
            let function = ordered(&device);
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            let mut first = function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap();
            let mut second = function.launch([Argument::Read(buffer.slice(..).unwrap())]).unwrap();
            crate::test::answer(&files[0], 2);
            assert_eq!(observe(&mut first), Poll::Pending);
            assert_eq!(observe(&mut second), Poll::Pending);
            assert!(matches!(
                crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]),
                Err(Error::Busy)
            ));
            for file in &files[1..] {
                crate::test::answer(file, 1);
            }
            assert_eq!(observe(&mut first), Poll::Ready(Ok(())));
            assert_eq!(observe(&mut second), Poll::Pending);
            assert!(matches!(
                crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]),
                Err(Error::Busy)
            ));
            for file in &files[1..] {
                crate::test::answer(file, 2);
            }
            assert_eq!(observe(&mut second), Poll::Ready(Ok(())));
            assert!(crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]).is_ok());
        }

        // Indirect memory remains leased until hardware completion after its observer is dropped.
        {
            let (device, file) = crate::test::running(&[0], 8);
            let function = crate::test::function(&device);
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            let alias = buffer.alias(..).unwrap();
            drop(
                function
                    .launch_holding([], &[(buffer.slice(..).unwrap(), crate::Access::Read)])
                    .unwrap(),
            );
            drop(buffer);
            assert!(matches!(
                crate::Lease::acquire([(alias.slice(..).unwrap(), crate::Access::Write)]),
                Err(Error::Busy)
            ));
            crate::test::answer(&file, 1);
            assert!(device.launches.complete(Contention::Skip));
            assert!(crate::Lease::acquire([(alias.slice(..).unwrap(), crate::Access::Write)]).is_ok());
        }

        // A transfer that retried a conflict with finished work still holds every view it reads.
        {
            let (device, file) = crate::test::running(&[0], 8);
            let function = crate::test::function(&device);
            let [mut x, mut y] = [(); 2].map(|()| device.alloc(Kind::Dram, 256).unwrap());
            drop(function.launch([y.slice_mut(..).unwrap().into()]).unwrap());
            crate::test::answer(&file, 1);
            let mut hosts = [(); 2].map(|()| crate::test::pinned());
            let [to_x, to_y] = hosts.each_mut().map(|host| host.slice_mut(..256));
            let mut read = Box::pin(device.read([(x.slice(..).unwrap(), to_x), (y.slice(..).unwrap(), to_y)]));
            assert!(read.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
            drop(read);
            assert_eq!(
                function.launch([x.slice_mut(..).unwrap().into()]).map(drop),
                Err(Error::Busy)
            );
        }

        // A multi-chip read into one pinned allocation holds it as a writer until every piece answers.
        {
            let device = crate::test::device(&[0, 1], 8);
            let buffer = device.alloc(Kind::Dram, 1024).unwrap();
            let mut host = crate::test::pinned();
            let host = host.slice_mut(..2048);
            let backing = host.backing.unwrap();
            let pairs = buffer.slice(..).unwrap().chips().zip(host.chunks(1024));
            let mut read = Box::pin(device.read(pairs));
            let published = read.as_mut().poll(&mut Context::from_waker(Waker::noop()));
            assert!(published.is_pending(), "{published:?}");
            assert!(matches!(backing.lease(Access::Read), Err(Error::Busy)));
        }

        // A cancelled large read keeps host writers exclusive until the driver answers.
        {
            let size = 48_usize << 30;
            let mut device = crate::test::device(&[0, 1, 2, 3, 4, 5, 6, 7], 8);
            device.allocations = Arc::new(crate::buffer::Allocations::over(
                device.topology(),
                crate::buffer::Ranges::new(0..size, size),
            ));
            let buffer = device.alloc(Kind::Dram, size).unwrap();
            let file = tempfile::tempfile().unwrap();
            file.set_len((size * 8) as u64).unwrap();
            let mut host = crate::Pinned::new(file.into(), size * 8).unwrap();
            let host = host.slice_mut(..);
            let backing = host.backing.unwrap();
            let mut transfer = Box::pin(device.read([(buffer.slice(..).unwrap(), host)]));
            assert!(
                transfer
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            drop(transfer);
            assert!(matches!(backing.lease(Access::Read), Err(Error::Busy)));
        }

        // Only a transport answer releases a transfer's memory; driver errors reach the observer.
        {
            use crate::driver::dma::fixture::{push, tag};
            use crate::driver::dma::{DmaQueue, Transport, fake::Fake};
            for fails in [false, true] {
                let buffer = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 1).unwrap();
                let host = crate::test::pinned();
                let fake = Fake::default();
                let queue = DmaQueue::new(Transport::Fake(fake.clone()), Arc::default(), Waker::noop().clone());
                let mut transfer = queue.acquire().await.unwrap();
                push(&mut transfer, &buffer, Some(&host));
                let tag = tag(&transfer);
                let mut observer = std::pin::pin!(transfer.submit());
                let mut context = Context::from_waker(Waker::noop());
                assert_eq!(observer.as_mut().poll(&mut context), Poll::Pending);
                let mut access = crate::Lease::new(1);
                let view = buffer.slice(..).unwrap();
                assert_eq!(access.extend([(view, Access::Write)]), Err(Error::Busy));

                let result = if fails {
                    Err(std::io::Error::from_raw_os_error(nix::libc::EIO))
                } else {
                    Ok(1)
                };
                fake.respond(tag, result);
                assert!(queue.complete());

                access.extend([(view, Access::Write)]).unwrap();
                let result = observer.as_mut().poll(&mut context);
                if fails {
                    assert!(matches!(result, Poll::Ready(Err(Error::Device(_)))));
                } else {
                    assert_eq!(result, Poll::Ready(Ok(())));
                }
            }
        }

        // A transfer holds its device and host memory until every answer, its observer dropped or
        // forgotten.
        {
            use crate::driver::dma::fixture::{activate, finish, push, tag};
            use crate::driver::dma::{DmaQueue, Transport, fake::Fake};
            for forget in [false, true] {
                for fails in [false, true] {
                    let buffer = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 1).unwrap();
                    let hosts = [crate::test::pinned(), crate::test::pinned()];
                    let queue = DmaQueue::new(Transport::Fake(Fake::default()), Arc::default(), Waker::noop().clone());
                    let mut transfer = queue.acquire().await.unwrap();
                    hosts.iter().for_each(|host| push(&mut transfer, &buffer, Some(host)));
                    let tag = tag(&transfer);
                    let mut observer = Box::pin(transfer.submit());
                    assert_eq!(
                        observer.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                        Poll::Pending
                    );
                    let mut access = crate::Lease::new(1);
                    let view = buffer.slice(..).unwrap();
                    activate(&queue, tag, fails.then(|| Error::Device("failed DMA".into())));
                    if forget {
                        std::mem::forget(observer);
                    } else {
                        drop(observer);
                    }
                    // One of two answers, or a duplicate of it, does not release the slot.
                    for _ in 0..1 + usize::from(fails) {
                        assert!(finish(&queue, tag + 1, Ok(())).is_none());
                        assert_eq!(access.extend([(view, Access::Write)]), Err(Error::Busy));
                    }
                    finish(&queue, tag, Ok(()));
                    access.extend([(view, Access::Write)]).unwrap();
                    assert_eq!(hosts[0][0], 0);
                    assert_eq!(queue.acquire().await.is_err(), fails);
                }
            }
        }
    }

    /// Indirect memory must belong to the launch device and cover all its chips.
    #[test]
    fn rejects_indirect_memory_from_another_device_or_partial_chips() {
        let device = Arc::new(crate::test::device(&[0, 1], 8));
        let foreign = crate::test::device(&[0, 1], 8).alloc(Kind::Dram, 256).unwrap();
        let buffer = device.alloc(Kind::Dram, 256).unwrap();
        let function = function(&device);
        assert!(matches!(
            function.launch_holding([], &[(foreign.slice(..).unwrap(), crate::Access::Read)]),
            Err(Error::ForeignBuffer)
        ));
        assert!(matches!(
            function.launch_holding([], &[(buffer.slice(..).unwrap().on(0).unwrap(), crate::Access::Read)]),
            Err(Error::Invalid(_))
        ));
        assert!(crate::Lease::acquire([(buffer.slice(..).unwrap(), crate::Access::Write)]).is_ok());
    }

    /// A refused launch holds neither a slot nor its operands.
    #[test]
    fn rejected_launches_release_their_slot() {
        let (device, file) = crate::test::running(&[0, 1], 8);
        let function = function(&device);
        let mut first = device.alloc(Kind::Dram, 256).unwrap();
        let mut second = device.alloc(Kind::Dram, 256).unwrap();
        assert!(matches!(
            function.launch([first.slice_mut(..).unwrap().on(0).unwrap().into()]),
            Err(Error::Invalid(_))
        ));
        let mut host = crate::test::pinned();
        assert!(matches!(
            function.launch([Argument::Host(host.slice_mut(..))]),
            Err(Error::Invalid(_))
        ));
        drop(function.launch([second.slice_mut(..).unwrap().into()]).unwrap());
        assert!(matches!(
            function.launch([first.slice_mut(..).unwrap().into(), second.slice(..).unwrap().into()]),
            Err(Error::Busy)
        ));
        let overflow = (0..9).map(|index| match index {
            ..7 => Argument::Value(0),
            _ => first.slice(..).unwrap().into(),
        });
        assert!(matches!(function.launch(overflow), Err(Error::Capacity)));
        drop(function.launch([first.slice_mut(..).unwrap().into()]).unwrap());
        crate::test::answer(&file, 2);
        assert!(device.launches.complete(Contention::Skip));

        // A refused indirect access releases operand leases before another launch retries.
        {
            let device = Arc::new(crate::test::device(&[0], 8));
            let function = crate::test::function(&device);
            let buffer = device.alloc(Kind::Dram, 256).unwrap();
            for _ in 0..2 {
                assert!(matches!(
                    function.launch_holding(
                        [crate::Argument::Write(buffer.slice(..).unwrap())],
                        &[(buffer.slice(..).unwrap(), crate::Access::Read)]
                    ),
                    Err(Error::Busy)
                ));
                assert!(crate::Lease::acquire([(buffer.slice(..).unwrap(), crate::Access::Write)]).is_ok());
            }
        }
    }

    /// A wait covers the conflicting work admitted before it: a read waits for writers, a write for
    /// every access. A transfer waits holding nothing, then moves its bytes after that work.
    #[test]
    fn waits_for_earlier_conflicting_work() {
        use std::os::unix::fs::FileExt;
        let (mut device, file) = crate::test::running(&[0], 8);
        let bar4 = tempfile::tempfile().unwrap();
        bar4.set_len(crate::driver::kernel::BAR4_DRAM + 4096).unwrap();
        let driver = Arc::get_mut(&mut Arc::get_mut(&mut device).unwrap().driver).unwrap();
        driver.chips[0].window = crate::driver::kernel::window::Window::new(bar4.try_clone().unwrap());
        let function = function(&device);
        let mut buffer = device.alloc(Kind::Dram, 256).unwrap();
        let at = crate::driver::kernel::BAR4_DRAM + buffer.addr() as u64;
        let mut context = Context::from_waker(Waker::noop());
        drop(function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
        let bytes = [7; 256];
        let mut write = Box::pin(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
        assert_eq!(write.as_mut().poll(&mut context), Poll::Pending);
        let mut landed = [0; 256];
        bar4.read_exact_at(&mut landed, at).unwrap();
        assert_eq!(landed, [0; 256]);
        crate::test::answer(&file, 1);
        device.launches.complete(Contention::Skip);
        assert_eq!(write.as_mut().poll(&mut context), Poll::Ready(Ok(())));
        bar4.read_exact_at(&mut landed, at).unwrap();
        assert_eq!(landed, bytes);
        drop(write);

        drop(function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
        let mut host = crate::test::pinned();
        // Each lease of the host memory retains its allocation.
        let allocation = Arc::clone(host.allocation());
        let mut read = Box::pin(device.read([(buffer.slice(..).unwrap(), host.slice_mut(..256))]));
        assert_eq!(read.as_mut().poll(&mut context), Poll::Pending);
        let leased = || Arc::strong_count(&allocation) > 2;
        assert!(!leased(), "a waiting read holds no host memory");
        crate::test::answer(&file, 2);
        device.launches.complete(Contention::Skip);
        assert_eq!(read.as_mut().poll(&mut context), Poll::Pending);
        assert!(leased(), "the read's DMA holds its host memory");

        // A read waits for overlapping writers on a shared chip, not for readers; a write waits for
        // both, and each wakes when the last record it waits for leaves.
        {
            use Access::{Read, Write};
            let buffer = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 256).unwrap();
            let wakes = Arc::new(Wakes::default());
            let (mut reader, mut writer) = (crate::Lease::new(1), crate::Lease::new(1));
            reader.extend([(buffer.slice(..128).unwrap(), Read)]).unwrap();
            assert_eq!(
                poll(&mut buffer.slice(..).unwrap().ready(Read), &wakes),
                Poll::Ready(Ok(()))
            );
            writer.extend([(buffer.slice(128..).unwrap(), Write)]).unwrap();
            for (bytes, chips, access, waits) in [
                (0..128, 1, Read, false),
                (128..256, 2, Read, false),
                (100..129, 1, Read, true),
                (0..64, 1, Write, true),
                (128..256, 1, Write, true),
            ] {
                let ready = poll(&mut buffer.slice(bytes).unwrap().selecting(chips).ready(access), &wakes);
                assert_eq!(ready.is_pending(), waits, "{access:?}");
            }
            let mut write = buffer.slice(..).unwrap().ready(Write);
            assert!(poll(&mut write, &wakes).is_pending());
            writer.clear();
            assert!(woken(&wakes) > 0);
            assert!(poll(&mut write, &wakes).is_pending());
            reader.clear();
            assert_eq!(poll(&mut write, &wakes), Poll::Ready(Ok(())));
        }

        // A wait covers the records admitted before it, not those admitted while it waits.
        {
            use Access::{Read, Write};
            let buffer = crate::test::allocations(0..512, 512).alloc(Kind::Dram, 256).unwrap();
            let wakes = Arc::new(Wakes::default());
            let (mut first, mut later) = (crate::Lease::new(1), crate::Lease::new(1));
            first.extend([(buffer.slice(..).unwrap(), Read)]).unwrap();
            let mut write = buffer.slice(..).unwrap().ready(Write);
            later.extend([(buffer.slice(..).unwrap(), Read)]).unwrap();
            assert!(poll(&mut write, &wakes).is_pending());
            first.clear();
            assert_eq!(poll(&mut write, &wakes), Poll::Ready(Ok(())));
            assert!(poll(&mut buffer.ready(Write), &wakes).is_pending());
        }
    }

    /// `synchronize` waits for the launches admitted before it, woken when they end, not for later
    /// ones, and fails once the device failed.
    #[test]
    fn synchronizes_admitted_launches() {
        let (device, file) = crate::test::running(&[0], 8);
        let function = function(&device);
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(Arc::clone(&wakes));
        let mut context = Context::from_waker(&waker);
        assert_eq!(
            Box::pin(device.synchronize()).as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        );
        drop(function.launch([]).unwrap());
        let mut first = Box::pin(device.synchronize());
        drop(function.launch([]).unwrap());
        assert_eq!(first.as_mut().poll(&mut context), Poll::Pending);
        crate::test::answer(&file, 1);
        device.launches.complete(Contention::Skip);
        assert_eq!(woken(&wakes), 1);
        assert_eq!(first.as_mut().poll(&mut context), Poll::Ready(Ok(())));
        let mut second = Box::pin(device.synchronize());
        assert_eq!(second.as_mut().poll(&mut context), Poll::Pending);
        let error = Error::Device("lost completion".into());
        device.driver.poison.set(error.clone());
        device.launches.complete(Contention::Skip);
        assert_eq!(second.as_mut().poll(&mut context), Poll::Ready(Err(error)));
        device.driver.stop();
    }

    /// The compiler's binder may bind each operand index to one storage only.
    #[test]
    fn binds_one_operand_per_storage() {
        let device = Arc::new(crate::test::device(&[0, 1], 8));
        let function = create(&device, 1, |args, bound| {
            for argument in args {
                bound.bind(0, argument)?;
            }
            Ok(std::future::ready(Ok(())))
        });
        let buffer = device.alloc(Kind::Dram, 512).unwrap();
        let other = device.alloc(Kind::Dram, 512).unwrap();
        let aliased = [buffer.slice(..).unwrap().into(), buffer.slice(..256).unwrap().into()];
        assert!(function.launch(aliased).is_ok());
        for view in [
            buffer.slice(256..).unwrap(),
            other.slice(..).unwrap(),
            buffer.slice(..).unwrap().on(0).unwrap(),
        ] {
            assert_eq!(
                function.launch([buffer.slice(..).unwrap().into(), view.into()]).err(),
                Some(Error::Invalid("operand 0 is bound to different storage".into()))
            );
        }
        assert!(function.launch([Argument::Value(1), Argument::Value(1)]).is_ok());
        assert_eq!(
            function.launch([Argument::Value(1), Argument::Value(2)]).err(),
            Some(Error::Invalid("operand 0 is bound to different storage".into()))
        );
        assert!(device.launches.complete(Contention::Skip));
    }

    struct Work {
        address: AtomicUsize,
        ready: Arc<AtomicBool>,
        completed: Arc<AtomicUsize>,
        _pin: PhantomPinned,
    }

    impl Work {
        fn new(ready: &Arc<AtomicBool>, completed: &Arc<AtomicUsize>) -> Self {
            Self {
                address: AtomicUsize::new(0),
                ready: Arc::clone(ready),
                completed: Arc::clone(completed),
                _pin: PhantomPinned,
            }
        }
    }

    impl Future for Work {
        type Output = Result<()>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            let address = std::ptr::from_ref(&*self).addr();
            let first = self
                .address
                .compare_exchange(0, address, Ordering::Relaxed, Ordering::Relaxed);
            assert!(first.is_ok() || first == Err(address));
            if self.ready.load(Ordering::Acquire) {
                self.completed.fetch_add(1, Ordering::Relaxed);
                Poll::Ready(Ok(()))
            } else {
                Poll::Pending
            }
        }
    }

    fn controlled(device: &Arc<Device>, ready: [Arc<AtomicBool>; 2], completed: Arc<AtomicUsize>) -> Function {
        create(device, 0, move |args, _| {
            let index = match args.next() {
                Some(Argument::Value(index)) => index as usize,
                None => 0,
                _ => panic!("call index"),
            };
            Ok(Work::new(&ready[index], &completed))
        })
    }

    /// A launch outlives its handles until its work and epilogue end, though its device goes with
    /// the last of them; a failure skips the epilogue.
    #[test]
    fn retains_active_work() {
        for failed in [false, true] {
            let device = Arc::new(crate::test::device(&[0], 1));
            let launches = Arc::clone(&device.launches);
            let allocations = Arc::clone(&device.allocations);
            let weak = Arc::downgrade(&device);
            let ready = Arc::new(AtomicBool::new(false));
            let completed = Arc::new(AtomicUsize::new(0));
            let function = {
                let (ready, completed, weak) = (Arc::clone(&ready), Arc::clone(&completed), weak.clone());
                create(&device, 1, move |args, bound| {
                    args.enumerate()
                        .try_for_each(|(index, argument)| bound.bind(index, argument))?;
                    let (work, device) = (Work::new(&ready, &completed), weak.upgrade());
                    Ok(async move {
                        assert!(!failed, "hardware failure must skip the epilogue");
                        drop(device);
                        work.await
                    })
                })
            };
            let input = device.alloc(Kind::Dram, 1).unwrap();
            let filler = device.alloc(Kind::Dram, device.usage(Kind::Dram).available).unwrap();
            let mut launch = function.launch([input.slice(..).unwrap().into()]).unwrap();
            drop(input);
            let error = Error::Device("lost completion".into());
            if failed {
                device.driver.poison.set(error.clone());
            }
            assert!(!launches.complete(Contention::Skip));
            assert_eq!(
                observe(&mut launch),
                if failed { Poll::Ready(Err(error)) } else { Poll::Pending }
            );
            assert_eq!(device.alloc(Kind::Dram, 1).is_err(), failed);
            assert_eq!(device.alloc(Kind::Sram, 1).is_err(), failed);
            if failed {
                assert!(function.launch([]).is_err());
                device.driver.stop();
            }
            drop((launch, function, device));
            assert!(weak.upgrade().is_none());
            if failed {
                drop(launches);
            } else {
                ready.store(true, Ordering::Release);
                assert!(launches.complete(Contention::Skip));
            }
            assert_eq!(completed.load(Ordering::Relaxed), usize::from(!failed));
            let usage = allocations.usage(Kind::Dram);
            assert_eq!(usage.available, allocations.capacity(Kind::Dram) - filler.len());
            assert!(allocations.alloc(Kind::Sram, 1).is_ok());
            drop(filler);
        }
    }

    /// An epilogue's error fails its own launch, not its function or the device.
    #[test]
    fn isolates_epilogue_failure() {
        let device = Arc::new(crate::test::device(&[0], 1));
        let bad = create(&device, 0, |args, bound| {
            args.enumerate()
                .try_for_each(|(index, argument)| bound.bind(index, argument))?;
            Ok(std::future::ready(Err(Error::Invalid("decode".into()))))
        });
        let good = function(&device);
        for _ in 0..=IN_FLIGHT {
            assert_eq!(
                observe(&mut bad.launch([]).unwrap()),
                Poll::Ready(Err(Error::Invalid("decode".into())))
            );
        }
        assert!(device.driver.poison.get().is_none());
        assert!(good.launch([]).is_ok());
        assert!(device.launches.complete(Contention::Skip));
    }

    /// A launch waits only for a free slot, and a receipt stays observable after its slot's reuse.
    #[test]
    fn waits_only_for_a_free_slot() {
        let device = Arc::new(crate::test::device(&[0], 1));
        let ready = [Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(true))];
        let function = Arc::new(controlled(&device, ready.clone(), Arc::default()));
        let mut first = function.launch([Argument::Value(0)]).unwrap();
        let mut completed = Vec::new();
        for _ in 0..IN_FLIGHT * 2 {
            let mut next = function.launch([Argument::Value(1)]).unwrap();
            assert_eq!(observe(&mut next), Poll::Ready(Ok(())));
            completed.push(next);
        }
        for _ in 1..IN_FLIGHT {
            drop(function.launch([Argument::Value(0)]).unwrap());
        }
        let (started, waiting) = std::sync::mpsc::channel();
        let (done, admitted) = std::sync::mpsc::channel();
        let thread = std::thread::spawn({
            let function = Arc::clone(&function);
            move || {
                started.send(()).unwrap();
                done.send(function.launch([Argument::Value(1)]).unwrap()).unwrap();
            }
        });
        waiting.recv().unwrap();
        assert!(admitted.recv_timeout(Duration::from_millis(10)).is_err());
        assert_eq!(observe(&mut first), Poll::Pending);
        ready[0].store(true, Ordering::Release);
        device.launches.complete(Contention::Skip);
        let mut last = admitted.recv_timeout(Duration::from_secs(1)).unwrap();
        thread.join().unwrap();
        for launch in completed.iter_mut().chain([&mut first, &mut last]) {
            assert_eq!(observe(launch), Poll::Ready(Ok(())));
        }
    }

    /// Host memory a launch uses is `Busy`; CPU access waits for it and fails after a failure.
    #[test]
    fn protects_host_access() {
        for failed in [false, true] {
            let (device, file) = crate::test::running(&[0], 1);
            let function = function(&device);
            let mut host = crate::test::mapped(&device, vec![0], 4096);
            let mut launch = function.launch([Argument::Host(host.slice_mut(..))]).unwrap();
            assert!(matches!(
                function.launch([Argument::Host(host.slice_mut(..))]),
                Err(Error::Busy)
            ));
            let (started, waiting) = std::sync::mpsc::channel();
            let (done, completed) = std::sync::mpsc::channel();
            let thread = std::thread::spawn(move || {
                started.send(()).unwrap();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| host[0] = 71));
                done.send(result.is_ok()).unwrap();
            });
            waiting.recv().unwrap();
            assert!(completed.recv_timeout(Duration::from_millis(10)).is_err());
            if failed {
                device.driver.poison.set(crate::Error::Device("lost completion".into()));
            } else {
                crate::test::answer(&file, 1);
            }
            device.launches.complete(Contention::Skip);
            assert_eq!(completed.recv_timeout(Duration::from_secs(1)).unwrap(), !failed);
            assert_eq!(observe(&mut launch).map(|result| result.is_ok()), Poll::Ready(!failed));
            thread.join().unwrap();
            device.driver.stop();
        }
    }

    /// Without firmware DMA a launch refuses host operands, and one whose task issued DMA fails,
    /// both with `Error::Unsupported`.
    #[test]
    fn refuses_device_dma_without_driver_support() {
        use furiosa_opt_ipc::{Answer, Failure, Status};
        let (device, file) = crate::test::unreserved(&[0], 8);
        let function = function(&device);
        let mut host = crate::test::pinned();
        assert_eq!(
            function.launch([Argument::Host(host.slice_mut(..))]).err(),
            Some(Error::Unsupported(crate::driver::NO_FIRMWARE_DMA.into()))
        );
        let mut launch = function.launch([]).unwrap();
        crate::test::respond::<()>(&file, &[(1, Answer(Err(Failure::new(Status::Internal, ""))))]);
        device.launches.complete(Contention::Skip);
        let failed = observe(&mut launch);
        assert_eq!(
            failed,
            Poll::Ready(Err(Error::Unsupported(crate::driver::NO_FIRMWARE_DMA.into())))
        );
    }
}
