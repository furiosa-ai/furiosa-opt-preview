//! What a profiled launch reports: the image names the span each marker opens or closes, every
//! chunk records the hits its selection holds with their cycle counts, and the host resolves them
//! into one span per executed command, or says how many hits a chunk could not keep.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::Poll;

use furiosa_opt_abi::image::{self, Context, Image, Kind, Level};
use furiosa_opt_ipc::Records;

use crate::registry::TARGET;
use crate::{Epilogue, Result};

mod selection;
mod spans;
pub use selection::Selection;
use spans::{Markers, Stamp, Window};

/// Rows a cluster's spans take on a timeline: its Task and trips, each context, the DMA queue,
/// and the command processor's own waits.
const ROWS: u64 = 8;

/// The profile a function's launches record, and the spans its image names.
pub(super) struct Profile {
    request: furiosa_opt_ipc::Profile,
    /// The kernel, as spans name it.
    name: String,
    spans: Vec<image::Span>,
    markers: Markers,
    /// Whether a lost hit was reported, which happens once a kernel.
    reported: AtomicBool,
}

impl Profile {
    /// The profile of `image` at `level`, recording what `selection` names in it, or `None` when
    /// the image names nothing to record. A selection the image does not hold records the Task only.
    pub(super) fn new(image: &Image<'_>, level: Level, selection: &Selection, name: String) -> Result<Option<Self>> {
        let profile = &image.profile;
        if profile.spans.is_empty() {
            return Ok(None);
        }
        if profile.unmarked > 0 {
            eprintln!(
                "furiosa-opt: {name} has {} commands without profile markers: the compiler ran out of marker ids",
                profile.unmarked
            );
        }
        let words = selection.resolve(profile).map_err(crate::Error::Invalid)?;
        let words = words.unwrap_or_else(|| {
            // An empty range: no marker but the Task's.
            furiosa_opt_ipc::Selection {
                range: 1 | 1 << 16,
                ..Default::default()
            }
        });
        let request = furiosa_opt_ipc::Profile::new(level as u8, words).expect("an image level is a profile level");
        Ok(Some(Self {
            request,
            name,
            spans: profile.spans.clone(),
            markers: Markers::new(&profile.spans),
            reported: AtomicBool::new(false),
        }))
    }

    /// Reports every executed command of `replies`, each cluster's chunks resolved together, and
    /// the hits a chunk could not keep.
    fn report(&self, replies: &[Records]) {
        for (cluster, chunks) in replies.iter().enumerate() {
            let hits = chunks.iter().flat_map(|chunk| &chunk.hits).copied().collect::<Vec<_>>();
            let attempted = chunks.iter().map(|chunk| u64::from(chunk.attempted)).sum::<u64>();
            let recorded = hits.len() as u64;
            if attempted > recorded {
                tracing::warn!(target: TARGET, kernel = self.name, cluster, recorded, attempted, "profile lost hits");
                if !self.reported.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "furiosa-opt: the profile of {} kept {recorded} of {attempted} hits on cluster {cluster}; \
                         select less with FURIOSA_OPT_PROFILE=<level>,<scope>[@<trips>]",
                        self.name
                    );
                }
            }
            for window in spans::resolve(&self.spans, &self.markers, &hits) {
                self.emit(cluster as u64, &window, recorded, attempted);
            }
        }
    }

    fn emit(&self, cluster: u64, window: &Window, recorded: u64, attempted: u64) {
        let span = &self.spans[window.span];
        let (row, kind) = match span.kind {
            Kind::Task => (0, "task"),
            Kind::Trip => (0, "trip"),
            Kind::Command(Context::Main) => (1, "main"),
            Kind::Command(Context::Sub) => (2, "sub"),
            Kind::Dma => (3, "dma"),
            Kind::Sync => (4, "sync"),
            Kind::Pcie => (4, "pcie"),
            Kind::Core => (5, "core"),
        };
        let bound = |stamp: Stamp| if stamp.exact { "exact" } else { "bound" };
        let task = span.kind == Kind::Task;
        tracing::info_span!(
            target: TARGET, "NPU", cat = "NPU", name = span.name, kernel = self.name,
            // A complete event carries its own extent, so a timeline reads device cycles, not the
            // host clock.
            ph = "Complete", ts = window.start.cycle,
            dur = window.end.cycle.saturating_sub(window.start.cycle),
            // A cluster counts its own cycles, so each is a set of rows.
            tid = cluster * ROWS + row,
            cluster, kind,
            issue_cycle = window.issue, begin_cycle = window.start.cycle, end_cycle = window.end.cycle,
            start_bound = bound(window.start), end_bound = bound(window.end),
            trip = window.trip, instruction_index = span.instruction.as_deref(),
            recorded = task.then_some(recorded), attempted = task.then_some(attempted),
        );
    }
}

/// A launch's epilogue. A profiled one asks every chunk for its function's profile and reports the
/// spans the replies resolve, in the span and subscriber the launch was made in.
pub(super) enum Report {
    Unprofiled,
    Profiled {
        profile: Arc<Profile>,
        span: tracing::Span,
        dispatch: tracing::Dispatch,
    },
}

impl Report {
    /// The epilogue of a launch made now, of a function loaded with `profile`.
    pub(super) fn new(profile: Option<&Arc<Profile>>) -> Self {
        match profile {
            Some(profile) => Self::Profiled {
                profile: Arc::clone(profile),
                span: tracing::Span::current(),
                dispatch: tracing::dispatcher::get_default(Clone::clone),
            },
            None => Self::Unprofiled,
        }
    }
}

impl Epilogue for Report {
    fn profile(&self) -> Option<furiosa_opt_ipc::Profile> {
        match self {
            Self::Profiled { profile, .. } => Some(profile.request),
            Self::Unprofiled => None,
        }
    }

    fn run(self: Pin<&mut Self>, replies: &[Records]) -> Poll<Result<()>> {
        let Self::Profiled {
            profile,
            span,
            dispatch,
        } = std::mem::replace(self.get_mut(), Self::Unprofiled)
        else {
            return Poll::Ready(Ok(()));
        };
        // The span ends inside its subscriber: a registry releases a closed span's parent through
        // the thread's default subscriber, which the worker running this does not set.
        tracing::dispatcher::with_default(&dispatch, move || span.in_scope(|| profile.report(replies)));
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use furiosa_opt_abi::executable::Executable;
    use furiosa_opt_abi::image::{DramKind, Parts, Slot, Stack};
    use furiosa_opt_ipc::Chunk;
    use tracing_subscriber::prelude::*;

    use super::*;
    use crate::Topology;
    use crate::memory::Kind as Memory;

    /// One cluster's image: the Task, a loop `3` whose body holds a main-context command `3.0`, and
    /// a DMA `4` after the loop.
    fn image() -> Image<'static> {
        let executable = Executable::encode(&vec![vec![&b"code"[..]; 2]; 1]).unwrap();
        let span = |kind, begin, end, name: &str, instruction: &str| image::Span {
            kind,
            begin,
            end,
            name: name.into(),
            instruction: Some(instruction.into()),
            idle: vec![],
        };
        Image::try_from(Parts {
            topology: Topology { chips: 1, pes: 8 },
            executable: Executable::parse(executable.leak()).unwrap(),
            weight: None,
            stacks: [Stack {
                kind: DramKind::Dram,
                size: 0,
            }; 2],
            slots: vec![Slot {
                kind: Memory::Dram,
                size: 8,
            }],
            inputs: vec![0],
            outputs: vec![],
            profile: image::Profile {
                spans: vec![
                    image::Span {
                        kind: Kind::Task,
                        begin: 0,
                        end: 1,
                        name: "Task".into(),
                        instruction: None,
                        idle: vec![],
                    },
                    span(Kind::Trip, 2, 5, "Trip", "3"),
                    span(Kind::Command(Context::Main), 3, 4, "TuExec", "3.0"),
                    span(Kind::Dma, 5, 6, "DMA", "4"),
                ],
                unmarked: 0,
            },
            sram_end: 0,
        })
        .unwrap()
    }

    fn profile(selection: &str) -> Profile {
        let selection = selection.parse().unwrap();
        Profile::new(&image(), Level::Trace, &selection, "kernel".into())
            .unwrap()
            .unwrap()
    }

    /// The fields of every span `report` made.
    fn reported(profile: &Profile, replies: &[Records]) -> Vec<BTreeMap<&'static str, String>> {
        #[derive(Clone, Default)]
        struct Recorded(Arc<Mutex<Vec<BTreeMap<&'static str, String>>>>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Recorded {
            fn on_new_span(
                &self,
                attributes: &tracing::span::Attributes<'_>,
                _: &tracing::Id,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                let mut fields = BTreeMap::new();
                attributes.record(&mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    fields.insert(field.name(), format!("{value:?}").trim_matches('"').to_string());
                });
                self.0.lock().unwrap().push(fields);
            }
        }
        let recorded = Recorded::default();
        let subscriber = tracing_subscriber::registry().with(recorded.clone());
        tracing::subscriber::with_default(subscriber, || profile.report(replies));
        recorded.0.lock().unwrap().clone()
    }

    fn hit(marker: u16, cycle: u64) -> u64 {
        u64::from(marker) | cycle << 16
    }

    /// A loop's trips select the markers of its body during those trips, the loop's trip marker,
    /// and nothing outside the loop.
    #[test]
    fn selects_trips_of_a_loop() {
        assert_eq!(
            profile("3@1..3").request.selection(),
            furiosa_opt_ipc::Selection {
                range: 3 | 5 << 16,
                trip: 2,
                first: 1,
                last: 2,
            }
        );
    }

    /// A scope without trips selects the markers of its instructions; one the image lacks selects
    /// only the Task; trips of an instruction that is no loop are refused.
    #[test]
    fn selects_the_markers_of_a_scope() {
        assert_eq!(profile("4").request.selection().range, 5 | 7 << 16);
        assert_eq!(profile("3..5").request.selection().range, 2 | 7 << 16);
        assert_eq!(profile("9").request.selection().range, 1 | 1 << 16);
        let selection = "3.0@1".parse().unwrap();
        assert!(Profile::new(&image(), Level::Trace, &selection, "kernel".into()).is_err());
    }

    /// Every span names its command's LIR index and kind, and the Task carries what the chunks
    /// recorded against what they tried to.
    #[test]
    fn reports_spans_with_their_command_and_losses() {
        let profile = profile("3");
        let hits = vec![hit(0, 0), hit(2, 10), hit(3, 11), hit(4, 12), hit(1, 99)];
        let chunk = Chunk { attempted: 7, hits };
        let spans = reported(&profile, &[vec![chunk]]);
        let named = spans
            .iter()
            .map(|span| (span["name"].as_str(), span["kind"].as_str()))
            .collect::<Vec<_>>();
        assert_eq!(named, [("Task", "task"), ("Trip", "trip"), ("TuExec", "main")]);
        assert_eq!(spans[2]["instruction_index"], "3.0");
        assert_eq!(
            (spans[0]["recorded"].as_str(), spans[0]["attempted"].as_str()),
            ("5", "7")
        );
    }

    /// The epilogue enters its launch's span where it runs.
    #[test]
    fn reports_in_the_launch_context() {
        use std::sync::atomic::AtomicBool;
        struct Context(Arc<AtomicBool>);

        impl<S> tracing_subscriber::Layer<S> for Context
        where
            S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
        {
            fn on_enter(&self, id: &tracing::span::Id, context: tracing_subscriber::layer::Context<'_, S>) {
                if context.span(id).is_some_and(|span| span.name() == "launch") {
                    self.0.store(true, Ordering::Relaxed);
                }
            }
        }

        let entered = Arc::new(AtomicBool::new(false));
        let subscriber = tracing_subscriber::registry().with(Context(Arc::clone(&entered)));
        let profile = Arc::new(profile("3"));
        let report = tracing::subscriber::with_default(subscriber, || {
            tracing::info_span!("launch").in_scope(|| Report::new(Some(&profile)))
        });
        assert_eq!(report.profile(), Some(profile.request));
        std::thread::spawn(move || {
            let mut report = report;
            let replies = [vec![Chunk {
                attempted: 2,
                hits: vec![hit(0, 0), hit(1, 9)],
            }]];
            assert_eq!(Pin::new(&mut report).run(&replies), Poll::Ready(Ok(())));
        })
        .join()
        .unwrap();
        assert!(entered.load(Ordering::Relaxed));
    }
}
