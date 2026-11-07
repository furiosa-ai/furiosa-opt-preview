//! On-device profile spans of kernels whose answer the hardware's order rules fix. A case runs in
//! a child process of its own, since a process reads `FURIOSA_OPT_PROFILE` once:
//!
//!   cargo furiosa-opt test -p furiosa-opt-examples --test profile_spans
#![cfg(backend = "npu")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use furiosa_opt_abi::image::{Image, Kind};
use furiosa_opt_examples::profiled::*;
use furiosa_opt_std::prelude::*;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

type Fields = BTreeMap<&'static str, String>;

/// Every span the runtime reports on `span::npu`, with all its fields.
#[derive(Clone, Default)]
struct Spans(Arc<Mutex<Vec<Fields>>>);

impl<S: tracing::Subscriber> Layer<S> for Spans {
    fn on_new_span(&self, attributes: &tracing::span::Attributes<'_>, _: &tracing::span::Id, _: Context<'_, S>) {
        struct Record<'a>(&'a mut Fields);
        impl Visit for Record<'_> {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &Field, value: &str) {
                self.0.insert(field.name(), value.to_string());
            }
        }
        if attributes.metadata().target() == "span::npu" {
            let mut fields = Fields::new();
            attributes.record(&mut Record(&mut fields));
            self.0.lock().unwrap().push(fields);
        }
    }
}

impl Spans {
    fn dump(&self) {
        if std::env::var_os("PROFILE_SPANS_DUMP").is_some() {
            for span in self.0.lock().unwrap().iter() {
                eprintln!("{span:?}");
            }
        }
    }

    fn of(&self, kind: &str) -> Vec<Fields> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|span| span["kind"] == kind)
            .cloned()
            .collect()
    }
}

fn cycle(span: &Fields, field: &str) -> u64 {
    span[field].parse().unwrap()
}

/// In the case's own process, a subscriber for its spans; otherwise `None`, once a child process
/// ran `case` with `FURIOSA_OPT_PROFILE=profile` and passed.
fn profiled(case: &str, profile: &str) -> Option<Spans> {
    if std::env::var_os("PROFILE_SPANS_CASE").is_some() {
        let spans = Spans::default();
        tracing_subscriber::registry().with(spans.clone()).init();
        return Some(spans);
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", case, "--nocapture"])
        .env("PROFILE_SPANS_CASE", case)
        .env("FURIOSA_OPT_PROFILE", profile)
        .status()
        .unwrap();
    assert!(status.success(), "{case} under FURIOSA_OPT_PROFILE={profile}");
    None
}

/// The LIR index of the one loop the image of kernel `path` can select trips of.
fn selectable_loop(path: &str) -> String {
    let entry = furiosa_opt_rt::registry::entries()
        .iter()
        .find(|entry| entry.path() == path)
        .expect("the kernel is in the registry");
    let image = Image::parse(entry.image()).unwrap();
    let mut loops = image.profile.spans.iter().filter(|span| span.kind == Kind::Trip);
    let index = loops.next().expect("a loop with markers").instruction.clone().unwrap();
    assert!(loops.next().is_none(), "one loop");
    index
}

/// A contraction's span is its own service on the context: two equal contractions take equal time,
/// and the second starts only after the first ended at the marker that follows its wait.
#[tokio::test]
async fn times_each_contraction_by_its_own_service() -> eyre::Result<()> {
    let Some(spans) = profiled("times_each_contraction_by_its_own_service", "trace") else {
        return Ok(());
    };
    let mut device = Device::new(contract_pair.topology())?;
    let input = HostTensor::<i8, m![AX, K8]>::from_vec(vec![1; <m![AX, K8]>::SIZE]);
    let weight = HostTensor::<i8, m![R, K8]>::from_vec(vec![1; <m![R, K8]>::SIZE]);
    let input = input.to_hbm::<m![1], m![AX, K8]>(&mut device.pdma).await?;
    let weight = weight.to_hbm::<m![1], m![R, K8]>(&mut device.pdma).await?;
    launch(contract_pair, (&mut device, &input, &weight)).await?;
    spans.dump();

    for cluster in ["0", "1"] {
        let pair = spans
            .of("main")
            .into_iter()
            .filter(|span| span["cluster"] == cluster)
            .collect::<Vec<_>>();
        let [first, second] = [&pair[0], &pair[1]];
        assert!(cycle(first, "end_cycle") <= cycle(second, "issue_cycle"), "{pair:?}");
        assert!(cycle(second, "begin_cycle") >= cycle(first, "end_cycle"), "{pair:?}");
        // The first runs after the context idled, which the firmware's clock throttle slows a little.
        let ratio = cycle(first, "dur") as f64 / cycle(second, "dur") as f64;
        assert!((0.8..1.25).contains(&ratio), "{pair:?}");
        // The two 4096-row contractions are most of the kernel's time.
        let task = spans
            .of("task")
            .into_iter()
            .find(|span| span["cluster"] == cluster)
            .unwrap();
        assert!(
            2 * (cycle(first, "dur") + cycle(second, "dur")) > cycle(&task, "dur"),
            "{pair:?} in {task:?}"
        );
    }
    Ok(())
}

/// A load queued behind another starts at that one's completion and takes as long.
#[tokio::test]
async fn starts_a_queued_load_at_the_completion_before_it() -> eyre::Result<()> {
    let Some(spans) = profiled("starts_a_queued_load_at_the_completion_before_it", "trace") else {
        return Ok(());
    };
    let mut device = Device::new(load_pair.topology())?;
    let host = HostTensor::<i32, m![PE, LOCAL, E2K]>::from_vec(vec![1; <m![PE, LOCAL, E2K]>::SIZE]);
    let first = host.to_hbm::<m![1], m![PE, LOCAL, E2K]>(&mut device.pdma).await?;
    let second = host.to_hbm::<m![1], m![PE, LOCAL, E2K]>(&mut device.pdma).await?;
    launch(load_pair, (&mut device, &first, &second)).await?;

    let transfers = spans.of("dma");
    let cluster = |span: &Fields| span["cluster"].clone();
    let loads = transfers
        .iter()
        .filter(|span| cluster(span) == "0")
        .take(2)
        .collect::<Vec<_>>();
    let [first, second] = [loads[0], loads[1]];
    assert_eq!(second["start_bound"], "exact", "{second:?}");
    assert_eq!(cycle(second, "begin_cycle"), cycle(first, "end_cycle"));
    let ratio = cycle(second, "dur") as f64 / cycle(first, "dur") as f64;
    assert!((0.75..1.25).contains(&ratio), "{first:?} against {second:?}");
    Ok(())
}

/// A selected trip records its own commands only, and every trip of its loop is timed.
#[tokio::test]
async fn records_the_commands_of_the_selected_trip() -> eyre::Result<()> {
    let index = selectable_loop("furiosa_opt_examples::profiled::rows");
    let profile = format!("trace,{index}@5");
    let Some(spans) = profiled("records_the_commands_of_the_selected_trip", &profile) else {
        return Ok(());
    };
    let mut device = Device::new(rows.topology())?;
    let input = HostTensor::<i8, m![TA, TB]>::from_vec(vec![1; <m![TA, TB]>::SIZE]);
    let input = input.to_hbm::<m![1], m![TA, TB]>(&mut device.pdma).await?;
    launch(rows, (&mut device, input.view())).await?;

    let on = |span: &Fields| span["cluster"] == "0";
    let trips = spans.of("trip").into_iter().filter(on).collect::<Vec<_>>();
    assert_eq!(trips.len(), 32);
    let transfers = spans.of("dma").into_iter().filter(on).collect::<Vec<_>>();
    assert_eq!(transfers.len(), 2, "{transfers:?}");
    assert!(
        transfers
            .iter()
            .all(|span| span.get("trip").is_some_and(|trip| trip == "5")),
        "{transfers:?}"
    );
    Ok(())
}

/// A trace past a chunk's queue keeps the hits that fit and says how many the code tried to record.
#[tokio::test]
async fn reports_the_hits_a_chunk_could_not_keep() -> eyre::Result<()> {
    let Some(spans) = profiled("reports_the_hits_a_chunk_could_not_keep", "trace") else {
        return Ok(());
    };
    let mut device = Device::new(many_rows.topology())?;
    let input = HostTensor::<i8, m![LA, LB]>::from_vec(vec![1; <m![LA, LB]>::SIZE]);
    let input = input.to_hbm::<m![1], m![LA, LB]>(&mut device.pdma).await?;
    launch(many_rows, (&mut device, input.view())).await?;

    for task in spans.of("task") {
        let (recorded, attempted) = (cycle(&task, "recorded"), cycle(&task, "attempted"));
        // The task's code queues at most 7,678 hits a chunk, then the Task's end.
        assert_eq!(recorded, 7679, "{task:?}");
        assert!(attempted > recorded, "{task:?}");
    }
    Ok(())
}
