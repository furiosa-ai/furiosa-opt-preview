//! What a profiled launch reports: the image names spans by the hits that begin and end them,
//! every chunk records the hits with their cycle counts, and the two resolve into cycle windows.

use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;

use furiosa_opt_abi::image::{self, Image, Level, MARKER_IDS};
use furiosa_opt_ipc::Records;

use crate::registry::TARGET;
use crate::{Epilogue, Error, Result};

/// The profile a function's launches ask every chunk to record, and the spans its image names.
pub(super) struct Profile {
    /// The level, and the ring each chunk keeps its hits in: an equal share of one reply.
    request: furiosa_opt_ipc::Profile,
    /// The kernel, as spans name it.
    name: String,
    markers: Vec<image::Span>,
}

impl Profile {
    /// The profile of `image` at `level`, or `None` when the image names nothing to record there.
    /// Refused when a chunk could fill its ring, which then wraps and reads as empty.
    pub(super) fn new(image: &Image<'_>, level: Level, name: String) -> Result<Option<Self>> {
        let depth = image.profile.depth(level);
        if depth == 0 || image.profile.spans.is_empty() {
            return Ok(None);
        }
        let ring = u32::try_from(image.executable.chunks())
            .ok()
            .filter(|&chunks| chunks <= furiosa_opt_ipc::Profile::CHUNKS)
            .and_then(|chunks| furiosa_opt_ipc::Profile::HITS.checked_div(chunks))
            .map_or(0, |share| share.min(furiosa_opt_ipc::Profile::QUEUE));
        if depth >= ring {
            return Err(Error::Invalid(format!(
                "{level:?} needs {depth} hits per chunk, but the profile ring holds {ring}"
            )));
        }
        let request = furiosa_opt_ipc::Profile::new(level as u8, ring).expect("a ring above the depth is a profile");
        Ok(Some(Self {
            request,
            name,
            markers: image.profile.spans.clone(),
        }))
    }

    /// Reports every completed span of `replies`, each cluster's chunks against their own markers:
    /// a begin hit opens and the next matching end closes, so a looped span yields one window per
    /// pass and an unmatched hit yields none.
    fn report(&self, replies: &[Records]) -> Result<()> {
        let ring = self.request.depth() as usize;
        if replies.iter().flatten().any(|hits| hits.len() >= ring) {
            // Arming a ring no chunk can fill makes this the device breaking the profile.
            return Err(Error::Device("a chunk filled its profile ring".into()));
        }
        for (cluster, records) in replies.iter().enumerate() {
            for (chunk, hits) in records.iter().enumerate() {
                let markers = self
                    .markers
                    .iter()
                    .filter(|marker| marker.chunk as usize == chunk && marker.cluster as usize == cluster);
                for marker in markers {
                    let mut begun = None;
                    for &raw in hits {
                        // A hit: its marker id in the low bits, its cycle count from bit 16 up.
                        let (id, cycle) = ((raw & u64::from(MARKER_IDS - 1)) as u16, raw >> 16);
                        match begun {
                            Some(begin) if marker.end == id => {
                                tracing::info_span!(
                                    target: TARGET, "NPU", cat = "NPU", name = marker.name,
                                    kernel = self.name,
                                    // A complete event carries its own extent, so a timeline reads
                                    // device cycles, not the host clock.
                                    ph = "Complete", ts = begin,
                                    // Hits keep the count's low 48 bits, so a span may cross a wrap.
                                    dur = cycle.wrapping_sub(begin) & (u64::MAX >> 16),
                                    // A cluster counts its own cycles, so each is a row.
                                    tid = cluster,
                                    begin_cycle = begin, end_cycle = cycle,
                                    instruction_index = marker.instruction.as_deref(),
                                    end_bound = if marker.upper_bound { "upper_bound" } else { "exact" },
                                );
                                begun = None;
                            }
                            None if marker.begin == id => begun = Some(cycle),
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(())
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
        Poll::Ready(tracing::dispatcher::with_default(&dispatch, move || {
            span.in_scope(|| profile.report(replies))
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use furiosa_opt_abi::executable::Executable;
    use furiosa_opt_abi::image::{DramKind, Parts, Slot, Stack};
    use tracing_subscriber::prelude::*;

    use super::*;
    use crate::Topology;
    use crate::memory::Kind;

    /// Spans on chunks 0 and 1 of cluster 0 and on chunk 0 of cluster 1, each opened by marker 1,
    /// in a ring of eight.
    fn profile() -> Profile {
        let marker = |chunk, cluster, name: &str, end| image::Span {
            chunk,
            cluster,
            name: name.into(),
            begin: 1,
            end,
            ..Default::default()
        };
        Profile {
            request: furiosa_opt_ipc::Profile::new(3, 8).unwrap(),
            name: "kernel".into(),
            markers: vec![
                marker(0, 0, "first", 2),
                marker(1, 0, "second", 3),
                marker(0, 1, "other", 2),
            ],
        }
    }

    /// A span `report` made: its name, its cluster, its cycle window and the duration it carries.
    #[derive(Debug, PartialEq, Eq)]
    struct Window {
        name: String,
        cluster: u64,
        begin: u64,
        end: u64,
        duration: u64,
    }

    fn reported(profile: &Profile, replies: &[Records]) -> Result<Vec<Window>> {
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
        tracing::subscriber::with_default(subscriber, || profile.report(replies))?;
        let window = |span: &BTreeMap<_, String>| {
            let field = |name| span[name].parse().unwrap();
            Window {
                name: span["name"].clone(),
                cluster: field("tid"),
                begin: field("begin_cycle"),
                end: field("end_cycle"),
                duration: field("dur"),
            }
        };
        Ok(recorded.0.lock().unwrap().iter().map(window).collect())
    }

    /// An image of `chunks` chunks on one chip's two clusters, naming one span per chunk that
    /// needs `named` hits at every level.
    fn image(chunks: usize, named: u32) -> Image<'static> {
        let executable = Executable::encode(&vec![vec![&b"code"[..]; 2]; chunks]).unwrap();
        Image::try_from(Parts {
            topology: Topology { chips: 1, pes: 8 },
            executable: Executable::parse(executable.leak()).unwrap(),
            weight: None,
            stacks: [Stack {
                kind: DramKind::Dram,
                size: 0,
            }; 2],
            slots: vec![Slot {
                kind: Kind::Dram,
                size: 8,
            }],
            inputs: vec![0],
            outputs: vec![],
            profile: image::Profile {
                depth: [named; 3],
                spans: (0..chunks as u32)
                    .map(|chunk| image::Span {
                        chunk,
                        name: "only".into(),
                        begin: 0,
                        end: 1,
                        ..Default::default()
                    })
                    .collect(),
            },
            sram_end: 0,
        })
        .unwrap()
    }

    /// A span pairs the hits of its own chunk and cluster; an unmatched hit yields nothing.
    #[test]
    fn spans_pair_within_chunk() {
        let replies = [vec![vec![1 | 100 << 16, 2 | 250 << 16], vec![1 | 300 << 16]]];
        let first = Window {
            name: "first".into(),
            cluster: 0,
            begin: 100,
            end: 250,
            duration: 150,
        };
        assert_eq!(reported(&profile(), &replies).unwrap(), [first]);
    }

    /// A span in a loop yields one window per pass.
    #[test]
    fn a_span_in_a_loop_yields_one_window_per_pass() {
        let hits = vec![
            2 | 50 << 16,
            1 | 100 << 16,
            2 | 250 << 16,
            1 | 300 << 16,
            2 | 450 << 16,
            1 | 500 << 16,
        ];
        let windows = reported(&profile(), &[vec![hits]]).unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| (window.begin, window.end))
                .collect::<Vec<_>>(),
            [(100, 250), (300, 450)]
        );
    }

    /// Spans name the cluster that recorded them, resolved against that cluster's markers.
    #[test]
    fn spans_name_the_cluster_that_recorded_them() {
        let reply = || vec![vec![1 | 100 << 16, 2 | 250 << 16]];
        let windows = reported(&profile(), &[reply(), reply()]).unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| (window.name.as_str(), window.cluster))
                .collect::<Vec<_>>(),
            [("first", 0), ("other", 1)]
        );
    }

    /// A span's duration counts across the wrap of the 48 cycle bits a hit keeps.
    #[test]
    fn times_a_span_across_the_cycle_wrap() {
        let wrap: u64 = 1 << 48;
        let replies = [vec![vec![1 | (wrap - 10) << 16, 2 | 5 << 16]]];
        assert_eq!(reported(&profile(), &replies).unwrap()[0].duration, 15);
    }

    /// Arming a ring no chunk can fill makes this the device breaking the profile, not an outcome
    /// a caller can meet.
    #[test]
    fn reports_a_chunk_that_filled_its_ring() {
        let filled = [vec![vec![1 | 100 << 16; 8]]];
        assert!(matches!(reported(&profile(), &filled), Err(Error::Device(_))));
    }

    /// The ring a chunk is armed with holds more than every hit the image names.
    #[test]
    fn arms_a_ring_no_chunk_can_fill() {
        let named = 270;
        let profile = Profile::new(&image(1, named), Level::Info, "kernel".into())
            .unwrap()
            .unwrap();
        assert!(
            profile.request.depth() > named,
            "a ring of {} holds no more",
            profile.request.depth()
        );
        assert_eq!(profile.request.level(), Level::Info as u8);
    }

    /// An image whose chunks share the reply evenly can name exactly the hits one chunk's ring
    /// holds, and that is the ring that wraps, so it is refused rather than armed.
    #[test]
    fn refuses_a_ring_an_image_could_fill_exactly() {
        let chunks = 2;
        let fills = furiosa_opt_ipc::Profile::HITS / chunks as u32;
        assert!(Profile::new(&image(chunks, fills), Level::Trace, "kernel".into()).is_err());
        assert!(
            Profile::new(&image(chunks, fills - 1), Level::Trace, "kernel".into())
                .is_ok_and(|profile| profile.is_some())
        );
    }

    /// The epilogue enters its launch's span where it runs, and ends it there, closing the
    /// parent the caller dropped first.
    #[test]
    fn reports_in_the_launch_context() {
        struct Context(Arc<AtomicBool>, Arc<AtomicBool>);

        impl<S> tracing_subscriber::Layer<S> for Context
        where
            S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
        {
            fn on_enter(&self, id: &tracing::span::Id, context: tracing_subscriber::layer::Context<'_, S>) {
                if context.span(id).is_some_and(|span| span.name() == "launch") {
                    self.0.store(true, Ordering::Relaxed);
                }
            }

            fn on_close(&self, id: tracing::span::Id, context: tracing_subscriber::layer::Context<'_, S>) {
                if context.span(&id).is_some_and(|span| span.name() == "call") {
                    self.1.store(true, Ordering::Relaxed);
                }
            }
        }

        let (entered, closed) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let subscriber = tracing_subscriber::registry().with(Context(Arc::clone(&entered), Arc::clone(&closed)));
        let profile = Arc::new(profile());
        let report = tracing::subscriber::with_default(subscriber, || {
            let launch = tracing::info_span!("call").in_scope(|| tracing::info_span!("launch"));
            launch.in_scope(|| Report::new(Some(&profile)))
        });
        assert_eq!(report.profile(), Some(profile.request));
        std::thread::spawn(move || {
            let mut report = report;
            let replies = [vec![vec![1 | 100 << 16, 2 | 250 << 16]]];
            assert_eq!(Pin::new(&mut report).run(&replies), Poll::Ready(Ok(())));
        })
        .join()
        .unwrap();
        assert!(entered.load(Ordering::Relaxed) && closed.load(Ordering::Relaxed));
    }
}
