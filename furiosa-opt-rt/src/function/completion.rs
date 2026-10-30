//! Every launch a device admitted, collected so that launches nobody observes still finish.

use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::Result;
use crate::device::Contention;
use crate::driver::mailbox::IN_FLIGHT;

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

    pub(super) fn push(&self, function: Arc<dyn Completion>) {
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

pub(super) trait Completion: Send + Sync {
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
