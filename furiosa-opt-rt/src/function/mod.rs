//! Eager execution with device-owned operands and the compiler's epilogues.

// allow: SIZE_OK — The slot lifecycle and its private-boundary tests share completion invariants.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use crate::buffer::Waiters;
use crate::driver::link::IN_FLIGHT;
use crate::driver::{Driver, Ticket};
use crate::memory::Kind;
use crate::{Access, Buffer, BufferSlice, Device, Error, Result};

mod argument;
mod table;
pub(crate) use argument::Operand;
pub use argument::{Argument, Arguments};
#[cfg(test)]
pub(crate) use table::Indirect;
use table::Table;
pub(crate) use table::{Inline, Prepared};

/// A function's compiled program in device DRAM.
pub(crate) struct Executable {
    pub(crate) device: Arc<Device>,
    pub(crate) staging: Arc<Buffer>,
    pub(crate) token: u64,
    pub(crate) sram_end: usize,
}

/// A loaded program that launches on its device with arguments its binder turns into operands.
pub struct Function {
    launch: Box<Callable>,
    ordered: bool,
}

type Callable = dyn for<'a> Fn(&mut dyn Iterator<Item = Argument<'a>>, &[(BufferSlice<'a>, Access)], bool) -> Result<Launch>
    + Send
    + Sync;

impl Function {
    /// Stages `bytes` on every chip and allocates slots for `capacity` arguments once.
    /// Launch temporaries hold SRAM below `sram_end`. `bind` turns a launch's arguments
    /// into operands and returns its epilogue, which runs only after the hardware finished; polling
    /// an epilogue must not block or launch a function, and its readback must use pinned host views.
    ///
    /// # Safety
    /// Binding must declare every caller-memory access with its mode; memory reached through an
    /// operand's contents must be declared through `launch_holding`.
    /// Device code must stay within those regions; writes to compiler storage need their own ordering proof.
    pub async unsafe fn load<F>(
        device: &Arc<Device>,
        bytes: &[u8],
        sram_end: usize,
        capacity: usize,
        bind: impl for<'a> Fn(&mut dyn Iterator<Item = Argument<'a>>, &mut Arguments) -> Result<F> + Send + Sync + 'static,
    ) -> Result<Self>
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let mut staging = device.alloc(Kind::Dram, bytes.len())?;
        device
            .write(std::iter::repeat(bytes.into()).zip(staging.slice_mut(..)?.chips()))
            .await?;
        let executable = Executable {
            device: Arc::clone(device),
            staging: Arc::new(staging),
            token: device.driver.token()?,
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
            worker: device.worker.waker.clone(),
            waiters: Mutex::default(),
        });
        device.worker.launches.push(slots.clone());
        Ok(Self {
            launch: Box::new(move |args, held, ordered| {
                slots.launch(&executable, held, |arguments| {
                    arguments.owner = if ordered {
                        crate::buffer::Owner::Ordered(executable.device.driver.owner()?)
                    } else {
                        crate::buffer::Owner::Unordered
                    };
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

impl<F: Future<Output = Result<()>> + Send + 'static> Slots<F> {
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
            slot.hardware = Some(self.submit(executable, &mut slot, held)?);
            slot.future = Some((future, Arc::clone(&executable.device.driver)));
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
            _device: Arc::clone(&executable.device),
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
    ) -> Result<Hardware> {
        let device = &executable.device;
        let chips = u32::from(device.topology().chips);
        for operand in &state.arguments.values {
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
                Operand::Host { mapping, .. } => {
                    if !mapping.table.belongs_to(&device.allocations) {
                        return Err(Error::ForeignBuffer);
                    }
                }
            }
        }
        for (view, _) in held {
            if !view.buffer.belongs_to(&device.allocations) {
                return Err(Error::ForeignBuffer);
            }
            if view.chips.count_ones() != chips {
                return Err(Error::Invalid("held memory must cover every chip of the device".into()));
            }
        }
        device.admit(|| state.arguments.acquire(held))?;
        let prepared = match Inline::new(device, &state.arguments.values) {
            Some(inline) => Prepared::Inline(inline),
            None => {
                let table = state.indirect.as_mut().expect("capacity allocates an indirect table");
                let bytes = table.encode(device, &state.arguments.values);
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
                Prepared::Indirect(table.upload(device, state.arguments.values.len())?)
            }
        };
        let ticket = self.wait(|| device.driver.submit(executable, &prepared))?;
        Ok(Hardware {
            ticket,
            _staging: Arc::clone(&executable.staging),
        })
    }

    fn advance(&self, slot: &Slot<F>) -> bool {
        try_lock(&slot.state).is_some_and(|state| self.step(slot, state))
    }

    /// Retires `slot`'s finished hardware, then its epilogue; true once the slot is idle.
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

struct Hardware {
    ticket: Ticket,
    _staging: Arc<Buffer>,
}

impl<F: Future<Output = Result<()>>> State<F> {
    /// The worker polls every active slot while anything runs, so epilogues need no waker.
    /// `running` falls once the hardware released the operands, before the epilogue runs.
    fn poll(&mut self, running: &AtomicBool) -> Poll<Result<()>> {
        let (future, driver) = self.future.as_mut().expect("active launch");
        if let Some(hardware) = &mut self.hardware {
            match driver.poll(&mut hardware.ticket) {
                Ok(false) => return Poll::Pending,
                Ok(true) => {
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
        unsafe { Pin::new_unchecked(future) }.poll(&mut Context::from_waker(Waker::noop()))
    }
}

/// One submitted launch of a [`Function`], resolving to the outcome of its hardware and epilogue.
/// Dropping it cancels nothing.
#[must_use = "await the launch to observe its result"]
pub struct Launch {
    completion: Arc<dyn Completion>,
    index: usize,
    generation: u64,
    _device: Arc<Device>,
}

trait Completion: Send + Sync {
    fn progress(&self) -> bool;
    /// Retires every launch the device finished, waiting out a poller retiring one now.
    fn retire(&self);
    fn poll(&self, index: usize, generation: u64, context: &mut Context<'_>) -> Poll<Result<()>>;
    /// The observer of launch `generation` in slot `index` is gone.
    fn release(&self, index: usize, generation: u64);
    /// The generation each slot launched last, unless every one of them ended.
    fn launched(&self) -> Option<[u64; IN_FLIGHT]>;
    /// Whether every launch of `launched` ended; `waker` is woken at the next end otherwise.
    fn ended(&self, launched: &[u64; IN_FLIGHT], waker: &Waker) -> bool;
}

/// How long a poll waits for a launch to finish before it returns `Pending`: a launch finishing
/// sooner is seen without a wake-up round trip through the executor.
const SPIN: Duration = Duration::from_micros(10);

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

/// Every function's launch slots on one device, polled by its worker so that launches nobody
/// observes still retire and run their epilogues.
#[derive(Default)]
pub(crate) struct Launches(Mutex<Vec<Arc<dyn Completion>>>);

/// Neither walk holds the list while it polls, since an epilogue's transfer may retire launches.
impl Launches {
    fn push(&self, function: Arc<dyn Completion>) {
        self.0.lock().unwrap().push(function);
    }

    /// Only the worker calls this, so a function it removes is the one at `index`.
    pub(crate) fn progress(&self) -> bool {
        let (mut idle, mut index) = (true, 0);
        while let Some(function) = self.get(index) {
            let complete = function.progress();
            idle &= complete;
            let mut functions = self.0.lock().unwrap();
            // The list and this clone: no Function or Launch holds the slots any more.
            if complete && Arc::strong_count(&function) == 2 {
                functions.remove(index);
            } else {
                index += 1;
            }
        }
        idle
    }

    /// Walks down: a concurrent removal shifts only later functions, so it skips none.
    pub(crate) fn retire(&self) {
        let mut index = self.0.lock().unwrap().len();
        while let Some(next) = index.checked_sub(1) {
            index = next;
            self.get(index).inspect(|function| function.retire());
        }
    }

    fn get(&self, index: usize) -> Option<Arc<dyn Completion>> {
        self.0.lock().unwrap().get(index).cloned()
    }

    /// The launches submitted so far that have not ended.
    pub(crate) fn launched(&self) -> Launched {
        Launched(
            (self.0.lock().unwrap().iter())
                .filter_map(|function| Some((Arc::clone(function), function.launched()?)))
                .collect(),
        )
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

struct Slots<F> {
    slots: Box<[Slot<F>]>,
    failure: OnceLock<Error>,
    worker: Waker,
    /// Woken whenever one of these launches ends.
    waiters: Mutex<Waiters>,
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
    hardware: Option<Hardware>,
    /// The epilogue, and the driver it waits on: the device itself goes with its last handle.
    future: Option<(F, Arc<Driver>)>,
    arguments: Arguments,
    indirect: Option<Table>,
    generation: u64,
    result: Result<()>,
    waker: Option<Waker>,
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

impl<F> Drop for State<F> {
    fn drop(&mut self) {
        if self.hardware.is_some() {
            self.arguments.poison();
        }
    }
}

impl<F: Future<Output = Result<()>> + Send + 'static> Completion for Slots<F> {
    fn progress(&self) -> bool {
        self.slots.iter().fold(true, |idle, slot| self.advance(slot) & idle)
    }

    /// An admission holds a slot that is not running, so this never waits for one.
    fn retire(&self) {
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

#[cfg(test)]
mod tests {
    use std::marker::PhantomPinned;

    use super::*;
    use arrayvec::ArrayVec;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    fn create<F: Future<Output = Result<()>> + Send + 'static>(
        device: &Arc<Device>,
        capacity: usize,
        bind: impl for<'a> Fn(&mut dyn Iterator<Item = Argument<'a>>, &mut Arguments) -> Result<F> + Send + Sync + 'static,
    ) -> Function {
        // SAFETY: stub executables do not access memory; these callbacks only exercise runtime ownership.
        let load = unsafe {
            Function::load(
                device,
                &[0],
                crate::arch::rngd::sram::WINDOW.end as usize,
                capacity,
                bind,
            )
        };
        match std::pin::pin!(load).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(function) => function.unwrap(),
            Poll::Pending => panic!("the fixture's small BAR4 write must finish synchronously"),
        }
    }

    fn function(device: &Arc<Device>) -> Function {
        create(device, 8, |args, bound| {
            args.enumerate()
                .try_for_each(|(index, argument)| bound.bind(index, argument))?;
            Ok(std::future::ready(Ok(())))
        })
    }

    fn observe(launch: &mut Launch) -> Poll<Result<()>> {
        Pin::new(launch).poll(&mut Context::from_waker(Waker::noop()))
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

    /// Compiled launches admit dependencies before completion, including aliases and indirect uses.
    #[test]
    fn admits_dependent_launches() {
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
                    assert!(device.worker.launches.progress());
                }
            }
        }
    }

    /// Peer-visible memory excludes conflicting compiled launches, including indirect uses.
    #[test]
    fn excludes_peer_conflicts() {
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

    /// Exhausted launch identities stay exhausted rather than reusing an earlier identity.
    #[test]
    fn rejects_exhausted_launches() {
        let (device, file) = crate::test::running(&[0], 8);
        let function = ordered(&device);
        device.driver.owners.store(u64::MAX - 1, Ordering::Relaxed);
        let mut first = function.launch([]).unwrap();
        for _ in 0..2 {
            assert!(matches!(function.launch([]), Err(Error::Capacity)));
        }
        crate::test::answer(&file, 1);
        assert_eq!(observe(&mut first), Poll::Ready(Ok(())));
    }

    /// Each queued launch excludes leases, and delays BAR4 writes, until its own completion, observer
    /// or not.
    #[test]
    fn retains_successor_access() {
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
        assert!(device.worker.launches.progress());
        assert!(crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Write)]).is_ok());
        let write = std::pin::pin!(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
        assert_eq!(write.poll(&mut Context::from_waker(Waker::noop())), Poll::Ready(Ok(())));
    }

    /// A queued dependency keeps its memory until every cluster answers that launch.
    #[test]
    fn retains_unanswered_clusters() {
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

    /// Generic tasks exclude both generic and compiled dependencies in either submission order.
    #[test]
    fn excludes_generic_tasks() {
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
            assert!(device.worker.launches.progress());
        }
    }

    /// Alternating operands reuse uploaded tables after completion as well as during overlap.
    #[test]
    fn reuses_alternating_tables() {
        use std::os::unix::fs::FileExt;

        for overlapped in [true, false] {
            // Given: two distinct indirect argument sets and explicitly controlled completion.
            let (mut device, files) = crate::test::clusters(8, 1 << 20);
            let driver = Arc::get_mut(&mut Arc::get_mut(&mut device).unwrap().driver).unwrap();
            let windows = driver
                .chips
                .iter_mut()
                .map(|chip| {
                    let bar4 = tempfile::tempfile().unwrap();
                    bar4.set_len(crate::arch::rngd::BAR4_DRAM + (1 << 20)).unwrap();
                    chip.window = crate::driver::window::Window::new(bar4.try_clone().unwrap());
                    bar4
                })
                .collect::<Vec<_>>();
            let function = create(&device, 128, |args, bound| {
                args.enumerate()
                    .try_for_each(|(index, argument)| bound.bind(index, argument))?;
                Ok(std::future::ready(Ok(())))
            });
            let mut cached = None;
            // When: A and B alternate, with the preceding launch either finished or still pending.
            for launch in 0..6_u32 {
                let args = (0..128).map(|index| Argument::Value(index + u64::from(launch % 2)));
                drop(function.launch(args).unwrap());
                let count = if overlapped { launch } else { launch + 1 };
                files.iter().for_each(|file| crate::test::answer(file, count));
                device.worker.launches.progress();
                let tables = windows
                    .iter()
                    .map(|window| {
                        let mut bytes = vec![0; 1 << 20];
                        window.read_exact_at(&mut bytes, crate::arch::rngd::BAR4_DRAM).unwrap();
                        bytes
                    })
                    .collect::<Vec<_>>();
                if launch == 1 {
                    cached = Some(tables);
                } else if launch > 1 {
                    assert!(
                        cached.as_ref().unwrap() == &tables,
                        "alternating cached operands rewrote device tables"
                    );
                }
            }
            files.iter().for_each(|file| crate::test::answer(file, 6));
            device.worker.launches.progress();
        }
    }

    /// Distinct ordered launches, including different functions, may share device writes.
    #[test]
    fn ordered_device_overlap() {
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
        device.worker.launches.progress();
    }

    /// Ordered host leases share one mapping while the earlier launch remains in flight.
    #[test]
    fn ordered_host_overlap() {
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
        device.worker.launches.progress();
    }

    /// A remapped host allocation cannot share ordered leases across publication domains.
    #[test]
    fn ordered_mapping_conflict() {
        let (device, file) = crate::test::running(&[0], 8);
        let (other, _) = crate::test::running(&[1], 8);
        // SAFETY: neither stub accesses caller memory.
        let first = unsafe { function(&device).ordered() };
        // SAFETY: neither stub accesses caller memory.
        let second = unsafe { function(&other).ordered() };
        let mut host = crate::test::mapped(&device, vec![0], 4096);
        drop(first.launch([Argument::Host(host.slice_mut(..))]).unwrap());
        host.mapping = crate::test::mapped(&other, vec![0], 4096).mapping;
        assert_eq!(
            second.launch([Argument::Host(host.slice_mut(..))]).err(),
            Some(Error::Busy)
        );
        crate::test::answer(&file, 1);
        device.worker.launches.progress();
        assert!(second.launch([Argument::Host(host.slice_mut(..))]).is_ok());
        other.driver.stop();
    }

    /// Only two ordered launches are exempt, in either submission order and memory kind.
    #[test]
    fn ordered_unordered_conflict() {
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
            device.worker.launches.progress();
        }
    }

    /// A transfer waits for a launch's HBM operands and is refused its mapped host operands; a launch
    /// is refused memory a transfer holds.
    #[test]
    fn ordered_transfer_conflict() {
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
                    device.worker.launches.progress();
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

    /// User leases and aliases within one ordered launch retain their existing exclusion.
    #[test]
    fn ordered_alias_conflict() {
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
        device.worker.launches.progress();
    }

    /// Hardware failure poisons ordered operands and rejects later ordered launches.
    #[test]
    fn ordered_failure_poison() {
        let (device, _) = crate::test::running(&[0], 8);
        // SAFETY: the stub has no memory accesses on any cluster.
        let function = unsafe { function(&device).ordered() };
        let buffer = device.alloc(Kind::Dram, 256).unwrap();
        let mut host = crate::test::mapped(&device, vec![0], 4096);
        drop(
            function
                .launch([
                    Argument::Write(buffer.slice(..).unwrap()),
                    Argument::Host(host.slice_mut(..)),
                ])
                .unwrap(),
        );
        let error = Error::Device("lost completion".into());
        device.driver.failure.set(error.clone()).unwrap();
        device.worker.launches.progress();
        assert_eq!(function.launch([]).err(), Some(error));
        assert_eq!(
            crate::Lease::acquire([(buffer.slice(..).unwrap(), Access::Read)]).err(),
            Some(Error::Poisoned)
        );
        assert_eq!(
            host.slice(..).backing.unwrap().lease(Access::Read).err(),
            Some(Error::Poisoned)
        );
        device.driver.stop();
    }

    /// Indirect memory remains leased until hardware completion after its observer is dropped.
    #[test]
    fn holds_indirect_memory_until_completion() {
        let (device, file) = crate::test::running(&[0], 8);
        let function = function(&device);
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
        assert!(device.worker.launches.progress());
        assert!(crate::Lease::acquire([(alias.slice(..).unwrap(), crate::Access::Write)]).is_ok());
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

    /// A refused indirect access releases operand leases before another launch retries.
    #[test]
    fn indirect_conflicts_release_operands() {
        let device = Arc::new(crate::test::device(&[0], 8));
        let function = function(&device);
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

    /// A launch, its completion and its slot's reuse allocate nothing.
    #[test]
    fn submits_without_allocations() {
        let device = Arc::new(crate::test::device(&[0], 1));
        let function = function(&device);
        let input = device.alloc(Kind::Dram, 1).unwrap();
        let mut launches = ArrayVec::<Launch, IN_FLIGHT>::new();
        crate::test::without_allocations(|| {
            for _ in 0..3 {
                for _ in 0..launches.capacity() {
                    launches.push(function.launch([input.slice(..).unwrap().into()]).unwrap());
                }
                device.worker.launches.progress();
                for launch in &mut launches {
                    assert_eq!(observe(launch), Poll::Ready(Ok(())));
                }
                launches.clear();
            }
        });
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
        assert!(device.worker.launches.progress());
    }

    /// A write to bytes a launch uses is `Busy` while the device runs it, and admitted, by a launch
    /// or a transfer, once the device finished it, though nobody observed it.
    #[test]
    fn admits_writes_after_unobserved_completion() {
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
    }

    /// A transfer of bytes a running launch writes waits for the launch, holding nothing, then
    /// moves them: a write lands in device memory, a read takes its host memory, only afterwards.
    #[test]
    fn transfers_wait_for_running_writers() {
        use std::os::unix::fs::FileExt;
        let (mut device, file) = crate::test::running(&[0], 8);
        let bar4 = tempfile::tempfile().unwrap();
        bar4.set_len(crate::arch::rngd::BAR4_DRAM + 4096).unwrap();
        let driver = Arc::get_mut(&mut Arc::get_mut(&mut device).unwrap().driver).unwrap();
        driver.chips[0].window = crate::driver::window::Window::new(bar4.try_clone().unwrap());
        let function = function(&device);
        let mut buffer = device.alloc(Kind::Dram, 256).unwrap();
        let at = crate::arch::rngd::BAR4_DRAM + buffer.addr() as u64;
        let mut context = Context::from_waker(Waker::noop());
        drop(function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
        let bytes = [7; 256];
        let mut write = Box::pin(device.write([(bytes.as_slice().into(), buffer.slice_mut(..).unwrap())]));
        assert_eq!(write.as_mut().poll(&mut context), Poll::Pending);
        let mut landed = [0; 256];
        bar4.read_exact_at(&mut landed, at).unwrap();
        assert_eq!(landed, [0; 256]);
        crate::test::answer(&file, 1);
        device.worker.launches.progress();
        assert_eq!(write.as_mut().poll(&mut context), Poll::Ready(Ok(())));
        bar4.read_exact_at(&mut landed, at).unwrap();
        assert_eq!(landed, bytes);
        drop(write);

        drop(function.launch([Argument::Write(buffer.slice(..).unwrap())]).unwrap());
        let mut host = crate::test::pinned();
        // Each lease of the host memory retains its allocation.
        let allocation = Arc::clone(&host.allocation);
        let mut read = Box::pin(device.read([(buffer.slice(..).unwrap(), host.slice_mut(..256))]));
        assert_eq!(read.as_mut().poll(&mut context), Poll::Pending);
        let leased = || Arc::strong_count(&allocation) > 2;
        assert!(!leased(), "a waiting read holds no host memory");
        crate::test::answer(&file, 2);
        device.worker.launches.progress();
        assert_eq!(read.as_mut().poll(&mut context), Poll::Pending);
        assert!(leased(), "the read's DMA holds its host memory");
    }

    /// `synchronize` waits for the launches admitted before it, woken when they end, not for later
    /// ones, and fails once the device failed.
    #[test]
    fn synchronizes_admitted_launches() {
        #[derive(Default)]
        struct Wakes(AtomicUsize);
        impl std::task::Wake for Wakes {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
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
        device.worker.launches.progress();
        assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
        assert_eq!(first.as_mut().poll(&mut context), Poll::Ready(Ok(())));
        let mut second = Box::pin(device.synchronize());
        assert_eq!(second.as_mut().poll(&mut context), Poll::Pending);
        let error = Error::Device("lost completion".into());
        device.driver.failure.set(error.clone()).unwrap();
        device.worker.launches.progress();
        assert_eq!(second.as_mut().poll(&mut context), Poll::Ready(Err(error)));
        device.driver.stop();
    }

    /// A transfer that retried a conflict with finished work still holds every view it reads, so
    /// a write to one is `Busy` while its unobserved DMA runs.
    #[test]
    fn holds_transfer_views_after_retried_conflict() {
        let (device, file) = crate::test::running(&[0], 8);
        let function = function(&device);
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
        assert!(device.worker.launches.progress());
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
            let launches = Arc::clone(&device.worker.launches);
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
                device.driver.failure.set(error.clone()).unwrap();
            }
            assert!(!launches.progress());
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
                assert!(launches.progress());
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
        assert!(device.driver.failure.get().is_none());
        assert!(good.launch([]).is_ok());
        assert!(device.worker.launches.progress());
    }

    /// A launch being bound never stalls the worker other launches complete through.
    #[test]
    fn progresses_during_binding() {
        let device = Arc::new(crate::test::device(&[0], 1));
        let binding = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(AtomicBool::new(false));
        let function = Arc::new({
            let (binding, completed) = (Arc::clone(&binding), Arc::clone(&completed));
            let ready = [Arc::new(AtomicBool::new(true)), Arc::clone(&ready)];
            create(&device, 0, move |args, _| {
                let Some(Argument::Value(index)) = args.next() else {
                    panic!("call index")
                };
                if index == 2 {
                    binding.store(true, Ordering::Release);
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while completed.load(Ordering::Acquire) < IN_FLIGHT {
                        assert!(Instant::now() < deadline, "binding blocked another call's epilogue");
                        std::thread::yield_now();
                    }
                }
                Ok(Work::new(&ready[usize::from(index != 0)], &completed))
            })
        });
        let mut first = function.launch([Argument::Value(0)]).unwrap();
        drop(function.launch([Argument::Value(1)]).unwrap());
        for _ in 2..IN_FLIGHT {
            drop(function.launch([Argument::Value(0)]).unwrap());
        }
        assert!(!device.worker.launches.progress());
        let thread = std::thread::spawn(move || function.launch([Argument::Value(2)]).unwrap());
        while !binding.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let observed = observe(&mut first);
        ready.store(true, Ordering::Release);
        device.worker.launches.progress();
        let next = thread.join().unwrap();
        assert_eq!(first.index, next.index);
        assert_eq!(observed, Poll::Ready(Ok(())));
        assert!(device.worker.launches.progress());
        assert_eq!(completed.load(Ordering::Relaxed), IN_FLIGHT + 1);
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
        device.worker.launches.progress();
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
                device
                    .driver
                    .failure
                    .set(crate::Error::Device("lost completion".into()))
                    .unwrap();
            } else {
                crate::test::answer(&file, 1);
            }
            device.worker.launches.progress();
            assert_eq!(completed.recv_timeout(Duration::from_secs(1)).unwrap(), !failed);
            assert_eq!(observe(&mut launch).map(|result| result.is_ok()), Poll::Ready(!failed));
            thread.join().unwrap();
            device.driver.stop();
        }
    }

    /// Without firmware DMA a launch refuses host operands, and one whose task issued DMA fails
    /// naming the missing driver support.
    #[test]
    fn refuses_device_dma_without_driver_support() {
        use furiosa_opt_ipc::{Response, Status};
        let missing = crate::driver::NO_FIRMWARE_DMA;
        let (device, file) = crate::test::unreserved(&[0], 8);
        let function = function(&device);
        let mut host = crate::test::pinned();
        assert_eq!(
            function.launch([Argument::Host(host.slice_mut(..))]).err(),
            Some(Error::Device(missing.into()))
        );
        let mut launch = function.launch([]).unwrap();
        let failure = Response::Error {
            status: Status::Internal as i32,
            message: String::new(),
        };
        crate::test::respond(&file, &[(1, failure)]);
        device.worker.launches.progress();
        let failed = observe(&mut launch);
        assert!(matches!(failed, Poll::Ready(Err(Error::Device(why))) if why.ends_with(missing)));
    }
}
