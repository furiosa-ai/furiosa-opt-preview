//! A cluster's hits as one window per executed command. A hit's marker opens or closes one span of
//! the image, and what the device stamped follows from the span's kind:
//!
//! - A tensor-unit command starts at its one marker, where the command processor passed it on an
//!   idle context. It ends no later than the next start on its context or the first marker a wait
//!   left that context idle at.
//! - A DMA ends at its completion and starts at its issue or at the completion before it on the
//!   cluster's queue, whichever is later, since the queue serves one transfer at a time in order.
//! - A synchronization or PCIe transfer holds the processor from its first marker to its second;
//!   the Task spans the launch; a trip runs to the next trip marker.

use std::collections::{BTreeMap, VecDeque};

use furiosa_opt_abi::image::{Context, Kind, Span};

/// One executed command: when the processor reached it, when it started and ended, and the trip of
/// the selected loop it was issued in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Window {
    pub(super) span: usize,
    pub(super) issue: u64,
    pub(super) start: Stamp,
    pub(super) end: Stamp,
    pub(super) trip: Option<u32>,
}

/// A cycle, and whether it is the event's own or a bound of it: no earlier for a start, no later
/// for an end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stamp {
    pub(super) cycle: u64,
    pub(super) exact: bool,
}

/// The span each marker id opens (`false`) or closes (`true`).
pub(super) struct Markers(BTreeMap<u16, (usize, bool)>);

impl Markers {
    pub(super) fn new(spans: &[Span]) -> Self {
        let edges = spans.iter().enumerate().flat_map(|(index, span)| match span.kind {
            Kind::Trip | Kind::Command(_) => vec![(span.begin, (index, false))],
            _ => vec![(span.begin, (index, false)), (span.end, (index, true))],
        });
        Self(edges.collect())
    }
}

/// The windows of one cluster's `hits`, oldest first, in the order `resolve` reads its issue.
pub(super) fn resolve(spans: &[Span], markers: &Markers, hits: &[u64]) -> Vec<Window> {
    let mut open: BTreeMap<usize, VecDeque<u64>> = BTreeMap::new();
    let mut pairs = vec![];
    let mut trips = vec![];
    for (marker, cycle) in unwrapped(hits) {
        match markers.0.get(&marker) {
            Some(&(span, false)) if spans[span].kind == Kind::Trip => trips.push((span, cycle)),
            Some(&(span, false)) if matches!(spans[span].kind, Kind::Command(_)) => {
                pairs.push((span, cycle, cycle));
            }
            Some(&(span, false)) => open.entry(span).or_default().push_back(cycle),
            Some(&(span, true)) => {
                // In-order completions pair a looped span's issue with the next completion.
                if let Some(issue) = open.get_mut(&span).and_then(VecDeque::pop_front) {
                    pairs.push((span, issue, cycle));
                }
            }
            None => {}
        }
    }
    pairs.sort_by_key(|&(_, issue, _)| issue);
    let last = pairs
        .iter()
        .map(|&(_, _, second)| second)
        .chain(trips.iter().map(|&(_, cycle)| cycle))
        .max();
    let mut windows = pairs
        .iter()
        .map(|&(span, issue, second)| {
            let at = |cycle| Stamp { cycle, exact: true };
            Window {
                span,
                issue,
                start: at(issue),
                end: at(second),
                trip: None,
            }
        })
        .collect::<Vec<_>>();
    commands(spans, &mut windows, last.unwrap_or(0));
    transfers(spans, &mut windows);
    let trip = |issue| {
        trips
            .iter()
            .rposition(|&(_, cycle)| cycle <= issue)
            .map(|trip| trip as u32)
    };
    for window in &mut windows {
        window.trip = trip(window.issue);
        // A trip's commands are done before the processor starts the next trip of the loop, so a
        // selected trip's last commands end no later than that trip's marker.
        if matches!(spans[window.span].kind, Kind::Command(_))
            && let Some(&(_, next)) = trips.iter().find(|&&(_, cycle)| cycle > window.issue)
            && next < window.end.cycle
        {
            window.end = Stamp {
                cycle: next,
                exact: false,
            };
        }
    }
    let task_end = windows
        .iter()
        .find(|window| spans[window.span].kind == Kind::Task)
        .map(|window| window.end);
    let periods = trips.iter().enumerate().map(|(index, &(span, cycle))| Window {
        span,
        issue: cycle,
        start: Stamp { cycle, exact: true },
        end: trips
            .get(index + 1)
            .map_or(task_end.unwrap_or(Stamp { cycle, exact: false }), |&(_, next)| Stamp {
                cycle: next,
                exact: true,
            }),
        trip: Some(index as u32),
    });
    windows.extend(periods);
    windows.sort_by_key(|window| window.issue);
    windows
}

/// Starts and ends of tensor-unit commands, walking each context's commands and the markers after
/// its waits in issue order; a command no later event bounds ends no later than `last`.
fn commands(spans: &[Span], windows: &mut [Window], last: u64) {
    for context in [Context::Main, Context::Sub] {
        let mut running: Option<usize> = None;
        for index in 0..windows.len() {
            let span = &spans[windows[index].span];
            if span.idle.contains(&context)
                && let Some(previous) = running.take()
            {
                windows[previous].end = Stamp {
                    cycle: windows[index].issue,
                    exact: false,
                };
            }
            if span.kind != Kind::Command(context) {
                continue;
            }
            // The end stays open until a later event on the context.
            let start = Stamp {
                cycle: windows[index].start.cycle,
                exact: false,
            };
            if let Some(previous) = running.replace(index) {
                windows[previous].end = start;
            }
        }
        if let Some(previous) = running {
            windows[previous].end = Stamp {
                cycle: last,
                exact: false,
            };
        }
    }
}

/// Starts of DMAs: each waits in the cluster's queue behind the completion before it.
fn transfers(spans: &[Span], windows: &mut [Window]) {
    let mut dmas = (0..windows.len())
        .filter(|&index| spans[windows[index].span].kind == Kind::Dma)
        .collect::<Vec<_>>();
    dmas.sort_by_key(|&index| windows[index].end.cycle);
    for pair in dmas.windows(2) {
        let (previous, index) = (windows[pair[0]].end.cycle, pair[1]);
        let window = &mut windows[index];
        window.start = Stamp {
            cycle: window.issue.max(previous),
            exact: true,
        };
    }
    // The first DMA may queue behind one the selection did not record.
    if let Some(&first) = dmas.first() {
        windows[first].start.exact = false;
    }
}

/// Each hit's marker id and cycle, the cycle carried past the 48-bit counter's wraps.
fn unwrapped(hits: &[u64]) -> impl Iterator<Item = (u16, u64)> + '_ {
    const PERIOD: u64 = 1 << 48;
    hits.iter().scan((0, 0), |(base, previous), &raw| {
        let cycle = raw >> 16;
        // Hits arrive in stamp order, completions within a few cycles of it, so a large step
        // back is the counter wrapping.
        if cycle + PERIOD / 2 < *previous {
            *base += PERIOD;
        }
        *previous = cycle;
        Some(((raw & 0xffff) as u16, *base + cycle))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kind: Kind, begin: u16, end: u16) -> Span {
        Span {
            kind,
            begin,
            end,
            name: format!("{kind:?}"),
            instruction: None,
            idle: vec![],
        }
    }

    fn waiting(span: Span, context: Context) -> Span {
        Span {
            idle: vec![context],
            ..span
        }
    }

    fn hit(marker: u16, cycle: u64) -> u64 {
        u64::from(marker) | cycle << 16
    }

    const MAIN: Kind = Kind::Command(Context::Main);

    fn windows(spans: &[Span], hits: &[u64]) -> Vec<(usize, u64, Stamp, Stamp)> {
        resolve(spans, &Markers::new(spans), hits)
            .into_iter()
            .map(|window| (window.span, window.issue, window.start, window.end))
            .collect()
    }

    fn exact(cycle: u64) -> Stamp {
        Stamp { cycle, exact: true }
    }

    fn bound(cycle: u64) -> Stamp {
        Stamp { cycle, exact: false }
    }

    /// A command ends no later than the next start on its context; the last one no later than the
    /// last stamp.
    #[test]
    fn ends_a_command_where_the_next_on_its_context_started() {
        let spans = [span(MAIN, 2, 2), span(MAIN, 4, 4)];
        let hits = [hit(2, 101), hit(4, 9_400)];
        assert_eq!(
            windows(&spans, &hits),
            [
                (0, 101, exact(101), bound(9_400)),
                (1, 9_400, exact(9_400), bound(9_400))
            ]
        );
    }

    /// A marker a wait left the command's context idle at ends it there, however late the next
    /// command on the context comes; one that leaves only the other context idle leaves it running.
    #[test]
    fn ends_a_command_at_the_marker_a_wait_left_its_context_idle_at() {
        let spans = [
            span(MAIN, 2, 2),
            waiting(span(Kind::Sync, 4, 5), Context::Sub),
            waiting(span(Kind::Dma, 6, 7), Context::Main),
            span(MAIN, 8, 8),
        ];
        let hits = [
            hit(2, 101),
            hit(4, 200),
            hit(5, 210),
            hit(6, 9_500),
            hit(8, 20_001),
            hit(7, 21_000),
        ];
        assert_eq!(windows(&spans, &hits)[0], (0, 101, exact(101), bound(9_500)));
    }

    /// A selected trip's last command, which no later event on its context bounds, ends no later
    /// than the next trip's marker.
    #[test]
    fn ends_a_trips_last_command_by_the_next_trip() {
        let spans = [span(Kind::Task, 0, 1), span(Kind::Trip, 2, 9), span(MAIN, 3, 3)];
        let hits = [hit(0, 0), hit(2, 10), hit(3, 12), hit(2, 50), hit(1, 900)];
        let command = windows(&spans, &hits).into_iter().find(|window| window.0 == 2).unwrap();
        assert_eq!(command.3, bound(50));
    }

    /// A DMA queued behind another starts at that one's completion, not at its own issue.
    #[test]
    fn starts_a_queued_dma_at_the_completion_before_it() {
        let spans = [span(Kind::Dma, 2, 3), span(Kind::Dma, 4, 5)];
        let hits = [hit(2, 100), hit(4, 110), hit(3, 8_300), hit(5, 16_400)];
        assert_eq!(
            windows(&spans, &hits),
            [
                (0, 100, bound(100), exact(8_300)),
                (1, 110, exact(8_300), exact(16_400))
            ]
        );
    }

    /// A span in a loop yields one window a pass, each issue paired with its own completion.
    #[test]
    fn pairs_each_pass_of_a_looped_span() {
        let spans = [span(Kind::Dma, 2, 3)];
        let hits = [hit(2, 10), hit(2, 20), hit(3, 30), hit(3, 40)];
        let ends = windows(&spans, &hits)
            .iter()
            .map(|window| (window.1, window.3.cycle))
            .collect::<Vec<_>>();
        assert_eq!(ends, [(10, 30), (20, 40)]);
    }

    /// Trip markers time each trip to the next one, the last to the Task's end, and every command
    /// is told the trip it was issued in.
    #[test]
    fn times_trips_and_numbers_the_commands_in_them() {
        let spans = [span(Kind::Task, 0, 1), span(Kind::Trip, 2, 9), span(Kind::Sync, 3, 4)];
        let hits = [hit(0, 0), hit(2, 10), hit(3, 12), hit(4, 15), hit(2, 50), hit(1, 90)];
        let resolved = resolve(&spans, &Markers::new(&spans), &hits);
        let trips = resolved
            .iter()
            .filter(|window| window.span == 1)
            .map(|window| (window.start.cycle, window.end.cycle, window.trip))
            .collect::<Vec<_>>();
        assert_eq!(trips, [(10, 50, Some(0)), (50, 90, Some(1))]);
        let sync = resolved.iter().find(|window| window.span == 2).unwrap();
        assert_eq!(sync.trip, Some(0));
    }

    /// A span's cycles count across the wrap of the 48 bits a hit keeps.
    #[test]
    fn times_a_span_across_the_cycle_wrap() {
        let wrap = 1u64 << 48;
        let spans = [span(Kind::Sync, 2, 3)];
        let resolved = windows(&spans, &[hit(2, wrap - 10), hit(3, 5)]);
        assert_eq!(resolved[0].3.cycle - resolved[0].2.cycle, 15);
    }
}
