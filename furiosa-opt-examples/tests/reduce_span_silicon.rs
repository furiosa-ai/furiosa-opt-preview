//! Runtime-span inter-slice reduces on the device: every launch bitwise against the static-span
//! twin and a host fold; device spans go to `RTGAT_SPANS`.
#![cfg(backend = "npu")]
use std::io::Write;
use std::sync::{Arc, Mutex};

use furiosa_opt_examples::reduce_span::{
    D, G, G5, G6, G7, G8, H, L5, L6, L7, L8, One, R, max_span, wave, wave_static_5, wave_static_6, wave_static_7,
    wave_static_8,
};
use furiosa_opt_std::prelude::*;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

#[derive(Default)]
struct Span {
    name: String,
    tid: u64,
    begin: u64,
    end: u64,
}

impl Visit for Span {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "name" && self.name.is_empty() {
            self.name = format!("{value:?}");
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "name" {
            value.clone_into(&mut self.name);
        }
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "begin_cycle" => self.begin = value,
            "end_cycle" => self.end = value,
            "tid" => self.tid = value,
            _ => {}
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_u64(field, value.cast_unsigned());
    }
}

#[derive(Clone, Default)]
struct Spans(Arc<Mutex<Vec<Span>>>);

impl<S: tracing::Subscriber> Layer<S> for Spans {
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _: &tracing::span::Id, _: LayerContext<'_, S>) {
        if attrs.metadata().target() == "span::npu" && attrs.metadata().name() == "NPU" {
            let mut span = Span::default();
            attrs.record(&mut span);
            self.0.lock().expect("spans").push(span);
        }
    }
}

struct Ledger {
    spans: Spans,
    file: Option<std::fs::File>,
    failures: Vec<String>,
}

impl Ledger {
    fn record(&mut self, kernel: &str, depth: i32, iter: usize) {
        let spans = std::mem::take(&mut *self.spans.0.lock().expect("spans"));
        let Some(file) = self.file.as_mut() else { return };
        for s in spans {
            writeln!(
                file,
                "{kernel}\t{depth}\t{iter}\t{}\t{}\t{}\t{}",
                s.name, s.tid, s.begin, s.end
            )
            .expect("ledger");
        }
    }

    fn check(&mut self, what: &str, depth: i32, iter: usize, got: &[f32], expected: &[f32]) {
        let bad = (0..got.len())
            .filter(|&i| got[i].to_bits() != expected[i].to_bits())
            .count();
        let first = (0..got.len()).find(|&i| got[i].to_bits() != expected[i].to_bits());
        println!("RESULT {what} depth={depth} iter={iter} bad={bad} first={first:?}");
        if let Some(i) = first {
            self.failures
                .push(format!("{what} depth={depth} iter={iter}: {bad} cells, first {i}"));
        }
    }
}

fn span_fold(input: &[f32], cells: usize, depth: usize, identity: f32, f: fn(f32, f32) -> f32) -> Vec<f32> {
    let span = 1 << depth.clamp(1, 8);
    (0..R::SIZE * cells)
        .map(|index| {
            let (slice, cell) = (index / cells, index % cells);
            let first = slice & !(span - 1);
            (first..first + span).fold(identity, |acc, source| f(acc, input[source * cells + cell]))
        })
        .collect()
}

fn per_slice(grouped: &[f32], cells: usize, depth: usize) -> Vec<f32> {
    (0..R::SIZE * cells)
        .map(|index| grouped[((index / cells) >> depth) * cells + index % cells])
        .collect()
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

#[tokio::test]
async fn reduce_span_on_silicon() -> eyre::Result<()> {
    let spans = Spans::default();
    tracing_subscriber::registry().with(spans.clone()).init();
    let mut ledger = Ledger {
        spans,
        file: std::env::var("RTGAT_SPANS")
            .ok()
            .map(|path| std::fs::File::create(path).expect("ledger")),
        failures: Vec::new(),
    };
    let iters: usize = env_or("RTGAT_ITERS", "6").parse().expect("RTGAT_ITERS");
    let depths: Vec<i32> = env_or("RTGAT_DEPTHS", "5,6,7,8")
        .split(',')
        .map(|d| d.parse().expect("RTGAT_DEPTHS"))
        .collect();

    let mut state = 11u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let scores_host: Vec<f32> = (0..R::SIZE * H::SIZE * G::SIZE).map(|_| uniform()).collect();
    let values_host: Vec<f32> = (0..R::SIZE * H::SIZE * G::SIZE * D::SIZE).map(|_| uniform()).collect();
    let (score_cells, value_cells) = (H::SIZE * G::SIZE, H::SIZE * G::SIZE * D::SIZE);
    let scores = || HostTensor::<f32, m![R, H, G]>::from_vec(scores_host.clone());
    let values = || HostTensor::<f32, m![R, H, G, D]>::from_vec(values_host.clone());

    let mut ctx = Device::new(wave.topology())?;
    let scores_hbm = scores().to_hbm(&mut ctx.pdma).await?;
    let values_hbm = values().to_hbm(&mut ctx.pdma).await?;

    macro_rules! static_case {
        ($kernel:ident, $depth:literal, $groups:ident, $lanes:ident) => {{
            let scores_hbm = HostTensor::<f32, m![$groups, $lanes, H, G]>::from_vec(scores_host.clone())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let values_hbm = HostTensor::<f32, m![$groups, $lanes, H, G, D]>::from_vec(values_host.clone())
                .to_hbm(&mut ctx.pdma)
                .await?;
            let mut reference = None;
            for iter in 0..iters {
                let (max, sum, pv) = launch($kernel, (&mut ctx, &scores_hbm, &values_hbm)).await?;
                ledger.record(stringify!($kernel), $depth, iter);
                let max = max.to_host::<m![$groups, H, G]>(&mut ctx.pdma).await?.into_vec();
                let sum = sum.to_host::<m![$groups, H, G]>(&mut ctx.pdma).await?.into_vec();
                let pv = pv.to_host::<m![$groups, H, G, D]>(&mut ctx.pdma).await?.into_vec();
                let result = (
                    per_slice(&max, score_cells, $depth),
                    per_slice(&sum, score_cells, $depth),
                    per_slice(&pv, value_cells, $depth),
                );
                let depth = $depth as usize;
                ledger.check(
                    "static-max-oracle",
                    $depth,
                    iter,
                    &result.0,
                    &span_fold(&scores_host, score_cells, depth, f32::MIN, f32::max),
                );
                ledger.check(
                    "static-sum-oracle",
                    $depth,
                    iter,
                    &result.1,
                    &span_fold(&scores_host, score_cells, depth, 0.0, |a, b| a + b),
                );
                ledger.check(
                    "static-pv-oracle",
                    $depth,
                    iter,
                    &result.2,
                    &span_fold(&values_host, value_cells, depth, 0.0, |a, b| a + b),
                );
                reference.get_or_insert(result);
            }
            ($depth, reference.unwrap())
        }};
    }
    let only = env_or("RTGAT_ONLY", "static,wave,max_span");
    let references = if only.contains("static") {
        let mut references = Vec::new();
        if depths.contains(&5) {
            references.push(static_case!(wave_static_5, 5, G5, L5));
        }
        if depths.contains(&6) {
            references.push(static_case!(wave_static_6, 6, G6, L6));
        }
        if depths.contains(&7) {
            references.push(static_case!(wave_static_7, 7, G7, L7));
        }
        if depths.contains(&8) {
            references.push(static_case!(wave_static_8, 8, G8, L8));
        }
        references
    } else {
        vec![]
    };

    for &depth in &depths {
        let reference = references.iter().find(|(d, _)| *d == depth).map(|(_, r)| r);
        let count = HostTensor::<i32, m![One]>::from_vec(vec![depth])
            .to_hbm(&mut ctx.pdma)
            .await?;
        let live = usize::try_from(depth).unwrap_or(0).clamp(1, 8);
        for iter in 0..iters {
            if only.contains("max_span") {
                let single = launch(max_span, (&mut ctx, &count, &scores_hbm)).await?;
                ledger.record("max_span", depth, iter);
                let single = single.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_vec();
                if let Some((static_max, _, _)) = reference {
                    ledger.check("mre-max-static", depth, iter, &single, static_max);
                }
                ledger.check(
                    "mre-max-oracle",
                    depth,
                    iter,
                    &single,
                    &span_fold(&scores_host, score_cells, live, f32::MIN, f32::max),
                );
            }
            if !only.contains("wave") {
                continue;
            }
            let (max, sum, pv) = launch(wave, (&mut ctx, &count, &scores_hbm, &values_hbm)).await?;
            ledger.record("wave", depth, iter);
            let max = max.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_vec();
            let sum = sum.to_host::<m![R, H, G]>(&mut ctx.pdma).await?.into_vec();
            let pv = pv.to_host::<m![R, H, G, D]>(&mut ctx.pdma).await?.into_vec();
            if let Some((static_max, static_sum, static_pv)) = reference {
                ledger.check("wave-max-static", depth, iter, &max, static_max);
                ledger.check("wave-sum-static", depth, iter, &sum, static_sum);
                ledger.check("wave-pv-static", depth, iter, &pv, static_pv);
            }
            ledger.check(
                "wave-max-oracle",
                depth,
                iter,
                &max,
                &span_fold(&scores_host, score_cells, live, f32::MIN, f32::max),
            );
            ledger.check(
                "wave-sum-oracle",
                depth,
                iter,
                &sum,
                &span_fold(&scores_host, score_cells, live, 0.0, |a, b| a + b),
            );
            ledger.check(
                "wave-pv-oracle",
                depth,
                iter,
                &pv,
                &span_fold(&values_host, value_cells, live, 0.0, |a, b| a + b),
            );
        }
    }
    let static_failures = ledger.failures.iter().filter(|f| f.contains("-static ")).count();
    println!("FAILURES {} (against static: {static_failures})", ledger.failures.len());
    for failure in &ledger.failures {
        println!("FAIL {failure}");
    }
    assert_eq!(static_failures, 0, "runtime span differs from its static twin");
    Ok(())
}
