//! Driver-channel submission and completion; descriptors remain owned by the transfer queue.

use std::collections::BTreeMap;

use arrayvec::ArrayVec;

use super::engine::Engine;
use super::{Channel, Command, Direction, Ring};
use crate::{Error, Result};

#[cfg_attr(
    test,
    expect(
        clippy::large_enum_variant,
        reason = "only tests add the small fake variant; production keeps the driver inline"
    )
)]
pub(crate) enum Transport {
    Driver {
        ring: Ring,
        channels: BTreeMap<u8, Channels>,
    },
    #[cfg(test)]
    Fake(fake::Fake),
}

impl Transport {
    pub(crate) fn open(chips: impl IntoIterator<Item = u8>, depth: u32) -> Result<Self> {
        let channels = chips
            .into_iter()
            .map(|chip| {
                let open = (0..4)
                    .map(|channel| Channel::open(chip, channel))
                    .collect::<Result<_>>()?;
                Ok((chip, Channels { open, next: [0; 2] }))
            })
            .collect::<Result<_>>()?;
        let ring = Ring::open(depth).map_err(|why| Error::Device(format!("opening io_uring: {why}")))?;
        Ok(Self::Driver { ring, channels })
    }
}

impl Engine for Transport {
    type Command = Command;

    /// Takes all of `commands` or, when the ring has no room for them, none.
    fn submit<'a>(&mut self, commands: impl Iterator<Item = (u64, &'a Command)>) -> Result<usize> {
        match self {
            Self::Driver { ring, channels } => {
                let entries = commands
                    .map(|(tag, command)| {
                        let channels = channels.get_mut(&command.chip()).expect("validated DMA chip");
                        (tag, command.encode(channels.select(command.direction()), tag))
                    })
                    .collect::<ArrayVec<_, { super::queue::DEPTH as usize }>>();
                ring.submit(entries.iter().map(|(tag, entry)| (*tag, entry)))
            }
            #[cfg(test)]
            Self::Fake(fake) => Ok(fake.submit(commands.map(|(tag, _)| tag))),
        }
    }

    fn complete(&mut self, done: impl FnMut(u64, Result<()>)) {
        match self {
            Self::Driver { ring, .. } => ring.complete(done),
            #[cfg(test)]
            Self::Fake(fake) => fake.complete(done),
        }
    }
}

pub(crate) struct Channels {
    open: Vec<Channel>,
    next: [usize; 2],
}

impl Channels {
    /// Each direction alternates between its two reserved channels.
    fn select(&mut self, direction: Direction) -> &Channel {
        let next = &mut self.next[direction as usize];
        *next ^= 1;
        &self.open[direction as usize * 2 + *next]
    }
}

#[cfg(test)]
pub(crate) mod fake {
    use std::collections::{BTreeSet, VecDeque};
    use std::io;
    use std::sync::{Arc, Mutex};

    use crate::{Error, Result};

    #[derive(Clone, Default)]
    pub(crate) struct Fake(Arc<Mutex<State>>);

    #[derive(Default)]
    struct State {
        pending: BTreeSet<u64>,
        answers: VecDeque<(u64, io::Result<i32>)>,
    }

    impl Fake {
        pub(super) fn submit(&mut self, tags: impl Iterator<Item = u64>) -> usize {
            let mut state = self.0.lock().unwrap();
            let mut count = 0;
            for tag in tags {
                assert!(state.pending.insert(tag), "a DMA command is published once");
                count += 1;
            }
            count
        }

        pub(crate) fn respond(&self, tag: u64, result: io::Result<i32>) {
            let mut state = self.0.lock().unwrap();
            assert!(state.pending.remove(&tag), "only a published DMA command can answer");
            state.answers.push_back((tag, result));
        }

        pub(super) fn complete(&mut self, mut done: impl FnMut(u64, Result<()>)) {
            for (tag, result) in self.0.lock().unwrap().answers.drain(..) {
                done(
                    tag,
                    result
                        .map(drop)
                        .map_err(|error| Error::Device(format!("DMA command failed: {error}"))),
                );
            }
        }
    }
}
