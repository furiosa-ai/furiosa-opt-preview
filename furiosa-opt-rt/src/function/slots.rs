//! One function's launch slots: their admission, completion and reuse.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError};
use std::task::{Context, Poll, Waker};

use furiosa_opt_ipc::Profile;

use super::completion::Completion;
use super::table::{Inline, Prepared};
use super::{Arguments, Epilogue, Executable, Launch, Operand, Table};
use crate::buffer::Waiters;
use crate::device::Contention;
use crate::driver::mailbox::IN_FLIGHT;
use crate::driver::{Driver, Replies, Ticket};
use crate::{Access, Device, Error, Hold, Result};

pub(super) struct Slots<F> {
    slots: Box<[Slot<F>]>,
    failure: OnceLock<Error>,
    worker: Waker,
    /// Woken whenever one of these launches ends.
    waiters: Mutex<Waiters>,
}

impl<F: Epilogue> Slots<F> {
    /// Slots for launches of `capacity` arguments each on `device`, whose worker runs their
    /// epilogues.
    pub(super) fn new(device: &Arc<Device>, capacity: usize) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
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
                            storage: Some(Box::new(None)),
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
        }))
    }

    pub(super) fn launch(
        self: &Arc<Self>,
        executable: &Executable,
        held: &[Hold<'_>],
        bind: impl FnOnce(&mut Arguments) -> Result<F>,
    ) -> Result<Launch> {
        let (index, mut slot) = self.wait(|| {
            Ok(self.slots.iter().enumerate().find_map(|(index, slot)| {
                let state = try_lock(&slot.state)?;
                // A failed launch keeps its slot while its observer may still ask for the error.
                let reusable = state.result.is_ok() || slot.released.load(Ordering::Acquire) >= state.generation;
                (state.hardware.is_none() && state.ended() && reusable).then_some((index, state))
            }))
        })?;
        let generation = slot.generation.checked_add(1).ok_or(Error::Capacity)?;
        let submitted = bind(&mut slot.arguments).and_then(|future| {
            slot.arguments.reserve(held.len());
            slot.hardware = Some(self.submit(executable, &mut slot, held, future.profile())?);
            let mut storage = slot.storage.take().expect("an ended slot holds its epilogue storage");
            *storage = Some(future);
            slot.future = Some((storage, Arc::clone(&executable.device.driver), Replies::new()));
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
        held: &[Hold<'_>],
        profile: Option<Profile>,
    ) -> Result<Ticket> {
        let device = &executable.device;
        let chips = u32::from(device.topology().chips);
        // Device memory a launch reaches is this device's, on every chip.
        let whole = |buffer: &crate::Buffer, covered: u32, what: &str| {
            if !buffer.belongs_to(&device.allocator) {
                return Err(Error::ForeignBuffer);
            }
            if covered != chips {
                return Err(Error::Invalid(format!("{what} must cover every chip of the device")));
            }
            Ok(())
        };
        for operand in state.arguments.operands() {
            match operand {
                Operand::Unbound | Operand::Value(_) => {}
                Operand::Device {
                    buffer,
                    chips: selected,
                    ..
                } => whole(buffer, selected.count_ones(), "a launch operand")?,
                Operand::Host { registration, .. } => {
                    if !registration.belongs_to(&device.allocator) {
                        return Err(Error::ForeignBuffer);
                    }
                }
            }
        }
        for view in held.iter().map(Hold::view) {
            whole(view.buffer(), view.chip_mask().count_ones(), "held memory")?;
        }
        // The clusters read the staged image while the launch runs.
        let image = executable.staging.slice(..).expect("a buffer holds its own range");
        device.admit(|| {
            state
                .arguments
                .acquire(held.iter().copied().chain([(image, Access::Read).into()]))
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
    fn step<'s>(&self, slot: &'s Slot<F>, mut state: MutexGuard<'s, State<F>>) -> bool {
        let mut ended = false;
        if state.future.is_some()
            && let Poll::Ready(result) = state.poll(&slot.running)
        {
            // The epilogue may hold the last device, whose worker stops only once it can lock this
            // slot: it drops unlocked, where it was pinned, before the launch ends.
            let (mut storage, driver, _) = state.future.take().expect("active launch");
            drop(state);
            *storage = None;
            drop(driver);
            state = slot.state.lock().unwrap();
            state.storage = Some(storage);
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
        let waker = state.ended().then(|| state.waker.take()).flatten();
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
        if state.ended() {
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
    future: Option<(Box<Option<F>>, Arc<Driver>, Replies)>,
    /// The slot's epilogue storage between launches, allocated once: an epilogue stays where its
    /// first poll pinned it until it drops there. Absent while a finished epilogue drops.
    storage: Option<Box<Option<F>>>,
    arguments: Arguments,
    indirect: Option<Table>,
    generation: u64,
    result: Result<()>,
    waker: Option<Waker>,
}

impl<F: Epilogue> State<F> {
    /// The last launch ended: its epilogue finished and dropped.
    fn ended(&self) -> bool {
        self.future.is_none() && self.storage.is_some()
    }

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
        let future = Option::as_mut(future).expect("an active launch's storage holds its epilogue");
        // SAFETY: the epilogue stays in its storage until it is dropped in place.
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
