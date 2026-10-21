//! The words between the host and one cluster's lead PE: frames out on the cluster's submission
//! ring, completions back on its completion ring.

use std::sync::Arc;

use super::kernel::map::Mapping;
use crate::arch::rngd::{self, DOORBELLS};
use crate::{Error, Result};
use furiosa_opt_abi::doorbell::Doorbells;
use furiosa_opt_abi::ring::{Completion, Consumer, Producer, Ring};

/// One cluster's submission and completion rings, the host's ends of both.
pub(crate) struct Link {
    submission: Producer<Arc<Mapping>>,
    completion: Consumer<Mapping>,
}

impl Link {
    /// Takes over doorbell `queue`'s rings in `bar2`, which maps the PE's doorbells from the first
    /// ring on, and the completion `page` of the cluster's node.
    pub(crate) fn new(bar2: Mapping, page: Mapping, queue: u8) -> Result<Self> {
        let why = |why| Error::Device(format!("the cluster's rings are not in a usable state: {why}"));
        let doorbells = Doorbells::new(Arc::new(bar2.starting_at(DOORBELLS.entries)), &DOORBELLS).map_err(why)?;
        Ok(Self {
            submission: doorbells.ring(queue).and_then(Producer::new).map_err(why)?,
            completion: Ring::completion(page, rngd::COMPLETION_RING)
                .and_then(Consumer::new)
                .map_err(why)?,
        })
    }

    /// The ring the host's requests go out on.
    pub(crate) fn submission(&mut self) -> &mut Producer<Arc<Mapping>> {
        &mut self.submission
    }

    /// The next completion the cluster published, if any.
    pub(crate) fn receive(&self) -> Result<Option<Completion>> {
        let mut words = [0; _];
        match (self.completion.copy(0, &mut words)).and_then(|()| self.completion.consume(words.len())) {
            Err(furiosa_opt_abi::ring::Error::Words) => return Ok(None),
            result => result.map_err(|error| Error::Device(error.to_string()))?,
        }
        let invalid = || Error::Device(format!("invalid completion header {:#x}", words[0]));
        Completion::decode(words).map(Some).ok_or_else(invalid)
    }
}
