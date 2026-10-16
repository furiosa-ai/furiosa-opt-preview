//! Runtime-row contractions on the device: each live prefix bitwise against the static twin and an
//! exact host oracle, the suffix against its sentinel; device spans go to `TUTIME_SPANS`.
#![cfg(backend = "npu")]
use std::io::Write;
use std::sync::{Arc, Mutex};

use furiosa_opt_examples::runtime_rows::runtime_rows;
use furiosa_opt_std::prelude::*;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

const SENTINEL: f32 = -9876.0;

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
    file: std::fs::File,
    failures: Vec<String>,
}

impl Ledger {
    fn record(&mut self, variant: &str, kind: &str, r: i32, iter: usize, host_us: u128) {
        let spans = std::mem::take(&mut *self.spans.0.lock().expect("spans"));
        writeln!(self.file, "{variant}\t{kind}\t{r}\t{iter}\thost\t-1\t0\t{host_us}").expect("ledger");
        for s in spans {
            writeln!(
                self.file,
                "{variant}\t{kind}\t{r}\t{iter}\t{}\t{}\t{}\t{}",
                s.name, s.tid, s.begin, s.end
            )
            .expect("ledger");
        }
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// Runs one variant: the static twin, then every count, each `iters` times.
macro_rules! variant {
    (
        $ledger:expr, $ctx:expr, $name:literal, [$($module:tt)*],
        $runtime:path, $static:path, $counts:expr, $iters:expr
    ) => {{
        use $($module)*::{One, Out, Red, Rows};
        let (rows, red, out) = (Rows::SIZE, Red::SIZE, Out::SIZE);
        let a: Vec<f32> = (0..rows * red).map(|i| f32::from(i16::try_from(i % 17).unwrap() - 8)).collect();
        let b: Vec<f32> = (0..out * red).map(|i| f32::from(i16::try_from(i % 11).unwrap() - 5)).collect();
        let oracle: Vec<f32> = (0..rows * out)
            .map(|i| {
                let dot = (0..red).map(|k| f64::from(a[i / out * red + k]) * f64::from(b[i % out * red + k]));
                dot.sum::<f64>() as f32
            })
            .collect();
        let input = HostTensor::<bf16, m![Rows, Red]>::from_vec(a.iter().map(|x| bf16::from_f32(*x)))
            .to_hbm::<m![1], m![Rows, Red]>(&mut $ctx.pdma)
            .await?;
        let weight = HostTensor::<bf16, m![Out, Red]>::from_vec(b.iter().map(|x| bf16::from_f32(*x)))
            .to_hbm::<m![1], m![Out, Red]>(&mut $ctx.pdma)
            .await?;
        let mut full = Vec::new();
        for (kind, counts) in [("static", vec![i32::try_from(rows).unwrap()]), ("runtime", $counts)] {
            for r in counts {
                let count = HostTensor::<i32, m![One]>::from_vec([r]).to_hbm::<m![1], m![One]>(&mut $ctx.pdma).await?;
                for iter in 0..$iters {
                    let mut output = HostTensor::<f32, m![Rows, Out]>::from_vec(vec![SENTINEL; rows * out])
                        .to_hbm::<m![1], m![Rows, Out]>(&mut $ctx.pdma)
                        .await?;
                    let start = std::time::Instant::now();
                    if kind == "static" {
                        launch($static, (&mut $ctx, &count, &input, &weight, &mut output)).await?;
                    } else {
                        launch($runtime, (&mut $ctx, &count, &input, &weight, &mut output)).await?;
                    }
                    let host_us = start.elapsed().as_micros();
                    $ledger.record($name, kind, r, iter, host_us);
                    let got = output.to_host::<m![Rows, Out]>(&mut $ctx.pdma).await?.into_vec();
                    let live = usize::try_from(r).unwrap_or(0).min(rows) * out;
                    let reference = if kind == "static" { &oracle } else { &full };
                    let first_bad = (0..live).find(|&i| got[i].to_bits() != reference[i].to_bits());
                    let tail_bad = (live..rows * out).find(|&i| got[i].to_bits() != SENTINEL.to_bits());
                    let oracle_bad = (0..live).find(|&i| got[i].to_bits() != oracle[i].to_bits());
                    println!(
                        "RESULT {} {kind} r={r} iter={iter} host_us={host_us} prefix_bad={first_bad:?} \
                         tail_bad={tail_bad:?} oracle_bad={oracle_bad:?}",
                        $name
                    );
                    if let Some(i) = first_bad.or(tail_bad).or(oracle_bad) {
                        $ledger.failures.push(format!(
                            "{} {kind} r={r} iter={iter}: index {i} (row {}, col {}) got {} bits {:#x}, \
                             static {:?}, oracle {}",
                            $name,
                            i / out,
                            i % out,
                            got[i],
                            got[i].to_bits(),
                            full.get(i),
                            oracle[i]
                        ));
                    }
                    if kind == "static" && iter == 0 {
                        full.clone_from(&got);
                    } else if kind == "static" && bits(&got) != bits(&full) {
                        $ledger.failures.push(format!("{} static iter={iter} differs from static iter=0", $name));
                    }
                }
            }
        }
    }};
}

#[tokio::test]
async fn runtime_rows_on_silicon() -> eyre::Result<()> {
    let spans = Spans::default();
    let _ = tracing_subscriber::registry().with(spans.clone()).try_init();
    let path = std::env::var("TUTIME_SPANS").unwrap_or_else(|_| "/tmp/opencode/tutime/spans.tsv".into());
    let iters: usize = std::env::var("TUTIME_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let only = std::env::var("TUTIME_ONLY").unwrap_or_default();
    let only: Vec<&str> = only.split(',').filter(|v| !v.is_empty()).collect();
    let want = |v: &str| only.is_empty() || only.contains(&v);
    let mut ledger = Ledger {
        spans,
        file: std::fs::File::create(&path)?,
        failures: Vec::new(),
    };
    let mut ctx = Device::new(runtime_rows.topology())?;
    let sweep = std::env::var("TUTIME_SWEEP").is_ok();
    let full: Vec<i32> = if sweep {
        (0..=129).collect()
    } else {
        vec![1, 7, 8, 64, 127, 128, 0, -1, 129]
    };
    let short: Vec<i32> = if sweep {
        (0..=33).collect()
    } else {
        vec![1, 7, 8, 16, 31, 32, 64, 127, 128, 0]
    };
    if want("mre") {
        variant!(
            ledger,
            ctx,
            "mre",
            [furiosa_opt_examples::runtime_rows],
            runtime_rows,
            furiosa_opt_examples::runtime_rows_silicon::mre::runtime_rows_static,
            full.clone(),
            iters
        );
    }
    if only.contains(&"mre-h") {
        variant!(
            ledger,
            ctx,
            "mre-h",
            [furiosa_opt_examples::runtime_rows],
            furiosa_opt_examples::runtime_rows_silicon::mre::runtime_rows_hoisted,
            furiosa_opt_examples::runtime_rows_silicon::mre::runtime_rows_static,
            full.clone(),
            iters
        );
    }
    if only.contains(&"panel-h") {
        variant!(
            ledger,
            ctx,
            "panel-h",
            [furiosa_opt_examples::runtime_rows_silicon::panel],
            furiosa_opt_examples::runtime_rows_silicon::panel::rows_hoisted,
            furiosa_opt_examples::runtime_rows_silicon::panel::rows_static,
            full.clone(),
            iters
        );
    }
    if want("panel") {
        variant!(
            ledger,
            ctx,
            "panel",
            [furiosa_opt_examples::runtime_rows_silicon::panel],
            furiosa_opt_examples::runtime_rows_silicon::panel::rows,
            furiosa_opt_examples::runtime_rows_silicon::panel::rows_static,
            full.clone(),
            iters
        );
    }
    if want("qkv") {
        variant!(
            ledger,
            ctx,
            "qkv",
            [furiosa_opt_examples::runtime_rows_silicon::qkv],
            furiosa_opt_examples::runtime_rows_silicon::qkv::rows,
            furiosa_opt_examples::runtime_rows_silicon::qkv::rows_static,
            short,
            iters
        );
    }
    for failure in &ledger.failures {
        println!("FAIL {failure}");
    }
    assert!(ledger.failures.is_empty(), "{} failing launches", ledger.failures.len());
    Ok(())
}

/// Two runtime-row contractions sharing one staged count, against two static ones: the per-command
/// part of the runtime cost is the difference to the single-contraction kernels.
#[tokio::test]
async fn twice_on_silicon() -> eyre::Result<()> {
    use furiosa_opt_examples::runtime_rows::{One, Out, Red, Rows};
    use furiosa_opt_examples::runtime_rows_silicon::mre::{runtime_rows_twice, runtime_rows_twice_static};
    let spans = Spans::default();
    let _ = tracing_subscriber::registry().with(spans.clone()).try_init();
    let path = std::env::var("TUTIME_SPANS").unwrap_or_else(|_| "/tmp/opencode/tutime/spans-twice.tsv".into());
    let iters: usize = std::env::var("TUTIME_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let mut ledger = Ledger {
        spans,
        file: std::fs::File::create(&path)?,
        failures: Vec::new(),
    };
    let mut ctx = Device::new(runtime_rows_twice.topology())?;
    let (rows, red, out) = (Rows::SIZE, Red::SIZE, Out::SIZE);
    let a: Vec<f32> = (0..rows * red)
        .map(|i| f32::from(i16::try_from(i % 17).unwrap() - 8))
        .collect();
    let b: Vec<f32> = (0..out * red)
        .map(|i| f32::from(i16::try_from(i % 11).unwrap() - 5))
        .collect();
    let oracle: Vec<f32> = (0..rows * out)
        .map(|i| {
            (0..red)
                .map(|k| f64::from(a[i / out * red + k]) * f64::from(b[i % out * red + k]))
                .sum::<f64>() as f32
        })
        .collect();
    let input = HostTensor::<bf16, m![Rows, Red]>::from_vec(a.iter().map(|x| bf16::from_f32(*x)))
        .to_hbm::<m![1], m![Rows, Red]>(&mut ctx.pdma)
        .await?;
    let weight = HostTensor::<bf16, m![Out, Red]>::from_vec(b.iter().map(|x| bf16::from_f32(*x)))
        .to_hbm::<m![1], m![Out, Red]>(&mut ctx.pdma)
        .await?;
    let mut full = Vec::new();
    for (kind, counts) in [("static", vec![128]), ("runtime", vec![1, 7, 64, 127, 128, 0, 129])] {
        for r in counts {
            let count = HostTensor::<i32, m![One]>::from_vec([r])
                .to_hbm::<m![1], m![One]>(&mut ctx.pdma)
                .await?;
            for iter in 0..iters {
                let mut first = HostTensor::<f32, m![Rows, Out]>::from_vec(vec![SENTINEL; rows * out])
                    .to_hbm::<m![1], m![Rows, Out]>(&mut ctx.pdma)
                    .await?;
                let mut second = HostTensor::<f32, m![Rows, Out]>::from_vec(vec![SENTINEL; rows * out])
                    .to_hbm::<m![1], m![Rows, Out]>(&mut ctx.pdma)
                    .await?;
                let start = std::time::Instant::now();
                if kind == "static" {
                    launch(
                        runtime_rows_twice_static,
                        (&mut ctx, &count, &input, &weight, &mut first, &mut second),
                    )
                    .await?;
                } else {
                    launch(
                        runtime_rows_twice,
                        (&mut ctx, &count, &input, &weight, &mut first, &mut second),
                    )
                    .await?;
                }
                let host_us = start.elapsed().as_micros();
                ledger.record("twice", kind, r, iter, host_us);
                let live = usize::try_from(r).unwrap_or(0).min(rows) * out;
                for (which, tensor) in [("first", first), ("second", second)] {
                    let got = tensor.to_host::<m![Rows, Out]>(&mut ctx.pdma).await?.into_vec();
                    let reference = if kind == "static" { &oracle } else { &full };
                    let bad = (0..live)
                        .find(|&i| {
                            got[i].to_bits() != reference[i].to_bits() || got[i].to_bits() != oracle[i].to_bits()
                        })
                        .or((live..rows * out).find(|&i| got[i].to_bits() != SENTINEL.to_bits()));
                    println!("RESULT twice {kind} r={r} iter={iter} {which} host_us={host_us} bad={bad:?}");
                    if let Some(i) = bad {
                        ledger.failures.push(format!(
                            "twice {kind} r={r} iter={iter} {which}: index {i} got {}",
                            got[i]
                        ));
                    }
                    if kind == "static" && iter == 0 && which == "first" {
                        full.clone_from(&got);
                    }
                }
            }
        }
    }
    for failure in &ledger.failures {
        println!("FAIL {failure}");
    }
    assert!(ledger.failures.is_empty(), "{} failing launches", ledger.failures.len());
    Ok(())
}
