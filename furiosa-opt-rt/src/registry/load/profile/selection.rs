//! The part of a launch a trace records: an LIR scope and, when the scope is a loop, some of its
//! trips. A load turns it into the words the device code tests, from the image's spans.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use furiosa_opt_abi::image::{Kind, Profile};

/// What a profiled launch records besides its Task: every marker, or those of `scope`, during
/// `trips` when the scope is a loop. Written `<scope>[@<trips>]`, as a span's `instruction_index`
/// prints the scope: `156.103` (an instruction, branch or loop), `244..297` (siblings), and a
/// trip `20` or trips `20..22`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    scope: Option<Scope>,
    trips: Option<Range<u32>>,
}

/// Sibling instructions `indices` in the block `parent` (empty for the top level) names.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Scope {
    parent: Vec<String>,
    indices: Range<usize>,
}

impl Selection {
    /// The device words of this selection in `profile`, or `None` when the image holds no marker
    /// of its scope.
    pub(super) fn resolve(&self, profile: &Profile) -> Result<Option<furiosa_opt_ipc::Selection>, String> {
        let Some(scope) = &self.scope else {
            return Ok(Some(furiosa_opt_ipc::Selection::default()));
        };
        let in_scope =
            |span: &&furiosa_opt_abi::image::Span| span.instruction.as_deref().is_some_and(|index| scope.holds(index));
        if let Some(trips) = &self.trips {
            let index = scope
                .single()
                .ok_or("trips select a loop, not a range of instructions")?;
            let Some(trip) = profile
                .spans
                .iter()
                .find(|span| span.kind == Kind::Trip && span.instruction.as_deref() == Some(&index))
            else {
                return match profile.spans.iter().any(|span| in_scope(&span)) {
                    true => Err(format!("{index} is not a loop")),
                    false => Ok(None),
                };
            };
            if trips.is_empty() {
                return Err(format!("no trip in {}..{}", trips.start, trips.end));
            }
            return Ok(Some(furiosa_opt_ipc::Selection {
                range: u32::from(trip.begin + 1) | u32::from(trip.end) << 16,
                trip: trip.begin,
                first: trips.start,
                last: trips.end - 1,
            }));
        }
        let markers = profile.spans.iter().filter(in_scope).flat_map(|span| match span.kind {
            Kind::Trip => vec![span.begin],
            _ => vec![span.begin, span.end],
        });
        let Some((low, high)) = markers.fold(None, |range, marker| match range {
            None => Some((marker, marker)),
            Some((low, high)) => Some((marker.min(low), marker.max(high))),
        }) else {
            return Ok(None);
        };
        Ok(Some(furiosa_opt_ipc::Selection {
            range: u32::from(low) | (u32::from(high) + 1) << 16,
            ..Default::default()
        }))
    }
}

impl Scope {
    /// Whether the instruction `index` names lies in this scope.
    fn holds(&self, index: &str) -> bool {
        let mut components = index.split('.');
        self.parent.iter().all(|part| components.next() == Some(part.as_str()))
            && components
                .next()
                .and_then(|position| position.parse().ok())
                .is_some_and(|position| self.indices.contains(&position))
    }

    /// The one instruction this scope names, as indices print it.
    fn single(&self) -> Option<String> {
        (self.indices.len() == 1).then(|| {
            let position = self.indices.start.to_string();
            self.parent
                .iter()
                .map(String::as_str)
                .chain([position.as_str()])
                .collect::<Vec<_>>()
                .join(".")
        })
    }
}

impl FromStr for Selection {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let (scope, trips) = match text.split_once('@') {
            Some((scope, trips)) => (scope, Some(range(trips)?)),
            None => (text, None),
        };
        // The scope's last component is an index or a range of them; the rest names its block.
        let (head, end) = match scope.split_once("..") {
            Some((head, end)) => (head, Some(end)),
            None => (scope, None),
        };
        let (parent, start) = head.rsplit_once('.').unwrap_or(("", head));
        let start = number(start)?;
        let end = end.map_or(Ok(start + 1), number)?;
        Ok(Self {
            scope: Some(Scope {
                parent: parent
                    .split('.')
                    .filter(|part| !part.is_empty())
                    .map(str::to_string)
                    .collect(),
                indices: start as usize..end as usize,
            }),
            trips,
        })
    }
}

/// `k` as `k..k + 1`, or `a..b`.
fn range(text: &str) -> Result<Range<u32>, String> {
    match text.split_once("..") {
        Some((start, end)) => Ok(number(start)?..number(end)?),
        None => number(text).map(|start| start..start + 1),
    }
}

fn number(text: &str) -> Result<u32, String> {
    text.parse().map_err(|_| format!("`{text}` is not a number"))
}

impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(scope) = &self.scope else {
            return write!(f, "every marker");
        };
        for part in &scope.parent {
            write!(f, "{part}.")?;
        }
        match scope.indices.len() {
            1 => write!(f, "{}", scope.indices.start)?,
            _ => write!(f, "{}..{}", scope.indices.start, scope.indices.end)?,
        }
        if let Some(trips) = &self.trips {
            match trips.len() {
                1 => write!(f, "@{}", trips.start)?,
                _ => write!(f, "@{}..{}", trips.start, trips.end)?,
            }
        }
        Ok(())
    }
}
