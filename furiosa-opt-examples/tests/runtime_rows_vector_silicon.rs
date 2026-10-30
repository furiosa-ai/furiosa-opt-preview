//! Runtime-row vector commands on the device: each live prefix bitwise against the static twin and
//! an exact host oracle, the suffix against its sentinel; device spans go to `TUTIME_SPANS`.
#![cfg(backend = "npu")]
use std::io::Write;
use std::sync::{Arc, Mutex};

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

    /// Checks one launch: the live prefix against the static run and the oracle, the rest unwritten.
    /// A command counting groups of `granule` rows writes every row of each group it reaches.
    fn check(
        &mut self,
        name: &str,
        kind: &str,
        r: i32,
        iter: usize,
        got: &[u32],
        reference: &[u32],
        oracle: &[u32],
        sentinel: u32,
        (row, granule): (usize, usize),
    ) {
        let live = usize::try_from(r).unwrap_or(0).div_ceil(granule) * granule;
        let live = live.min(got.len() / row) * row;
        let prefix_bad = (0..live).find(|&i| got[i] != reference[i]);
        let oracle_bad = (0..live).find(|&i| got[i] != oracle[i]);
        let tail_bad = (live..got.len()).find(|&i| got[i] != sentinel);
        println!(
            "RESULT {name} {kind} r={r} iter={iter} prefix_bad={prefix_bad:?} tail_bad={tail_bad:?} oracle_bad={oracle_bad:?}"
        );
        if let Some(i) = prefix_bad.or(oracle_bad).or(tail_bad) {
            self.failures.push(format!(
                "{name} {kind} r={r} iter={iter}: index {i} (row {}) got {:#x}",
                i / row,
                got[i]
            ));
        }
    }
}

/// Runs one variant: the static twin, then every count, each `iters` times.
macro_rules! variant {
    ($ledger:expr, $ctx:expr, $name:literal, $d:ty, [$($module:tt)*], $runtime:path, $static:path,
     $sentinel:expr, $to:expr, $from:expr, $oracle:expr, $counts:expr, $iters:expr) => {
        variant!($ledger, $ctx, $name, $d, [$($module)*], $runtime, $static, $sentinel, $to, $from,
            $oracle, $counts, $iters, 1)
    };
    ($ledger:expr, $ctx:expr, $name:literal, $d:ty, [$($module:tt)*], $runtime:path, $static:path,
     $sentinel:expr, $to:expr, $from:expr, $oracle:expr, $counts:expr, $iters:expr, $granule:expr) => {{
        use $($module)*::{H, One, Rows};
        let (rows, h) = (Rows::SIZE, H::SIZE);
        // Small integers, so every sum is exact in bf16 and the oracle needs no rounding rule.
        let a: Vec<f32> = (0..rows * h).map(|i| f32::from(u16::try_from((i * 7 + 3 * (i / h)) % 16).unwrap())).collect();
        let b: Vec<f32> = (0..rows * h).map(|i| f32::from(u16::try_from((i * 13 + i / h) % 16).unwrap())).collect();
        let oracle: Vec<u32> = a.iter().zip(&b).map($oracle).collect();
        let lhs = HostTensor::<$d, m![Rows, H]>::from_vec(a.iter().map($to)).to_hbm::<m![1], m![Rows, H]>(&mut $ctx.pdma).await?;
        let rhs = HostTensor::<$d, m![Rows, H]>::from_vec(b.iter().map($to)).to_hbm::<m![1], m![Rows, H]>(&mut $ctx.pdma).await?;
        let mut full = Vec::new();
        for (kind, counts) in [("static", vec![i32::try_from(rows).unwrap()]), ("runtime", $counts)] {
            for r in counts {
                let count = HostTensor::<i32, m![One]>::from_vec([r]).to_hbm::<m![1], m![One]>(&mut $ctx.pdma).await?;
                for iter in 0..$iters {
                    let mut output = HostTensor::<$d, m![Rows, H]>::from_vec(vec![$sentinel; rows * h])
                        .to_hbm::<m![1], m![Rows, H]>(&mut $ctx.pdma)
                        .await?;
                    let start = std::time::Instant::now();
                    if kind == "static" {
                        launch($static, (&mut $ctx, &count, &lhs, &rhs, &mut output)).await?;
                    } else {
                        launch($runtime, (&mut $ctx, &count, &lhs, &rhs, &mut output)).await?;
                    }
                    $ledger.record($name, kind, r, iter, start.elapsed().as_micros());
                    let got: Vec<u32> = output.to_host::<m![Rows, H]>(&mut $ctx.pdma).await?.into_vec().iter().map($from).collect();
                    if kind == "static" && iter == 0 {
                        full.clone_from(&got);
                    }
                    let sentinel = [$sentinel].iter().map($from).next().unwrap();
                    $ledger.check($name, kind, r, iter, &got, &full, &oracle, sentinel, (h, $granule));
                }
            }
        }
    }};
}

#[tokio::test]
async fn vector_rows_on_silicon() -> eyre::Result<()> {
    use furiosa_opt_examples::runtime_rows_vector::runtime_rows_add;
    use furiosa_opt_examples::runtime_rows_vector_silicon::{add, grouped, tokenwise, wide};
    let spans = Spans::default();
    let _ = tracing_subscriber::registry().with(spans.clone()).try_init();
    let path = std::env::var("TUTIME_SPANS").unwrap_or_else(|_| "/tmp/opencode/rtmain/spans.tsv".into());
    let iters: usize = std::env::var("TUTIME_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let only = std::env::var("TUTIME_ONLY").unwrap_or_default();
    let want = |v: &str| only.is_empty() || only.split(',').any(|o| o == v);
    let mut ledger = Ledger {
        spans,
        file: std::fs::File::create(&path)?,
        failures: Vec::new(),
    };
    let mut ctx = Device::new(runtime_rows_add.topology())?;
    let counts = || vec![1, 8, 64, 127, 128, 7, 9, 0, -1, 129];
    let bf16_sum = |(x, y): (&f32, &f32)| {
        u32::from(bf16::from_f32(bf16::from_f32(*x).to_f32() + bf16::from_f32(*y).to_f32()).to_bits())
    };
    if want("add") {
        variant!(
            ledger,
            ctx,
            "add",
            f32,
            [furiosa_opt_examples::runtime_rows_vector],
            runtime_rows_add,
            add::add_static,
            -9876.0f32,
            |x: &f32| *x,
            |x: &f32| x.to_bits(),
            |(x, y): (&f32, &f32)| (x + y).to_bits(),
            counts(),
            iters
        );
    }
    if want("wide") {
        variant!(
            ledger,
            ctx,
            "wide",
            bf16,
            [furiosa_opt_examples::runtime_rows_vector_silicon::wide],
            wide::rows,
            wide::rows_static,
            bf16::from_f32(-9876.0),
            |x: &f32| bf16::from_f32(*x),
            |x: &bf16| u32::from(x.to_bits()),
            bf16_sum,
            counts(),
            iters
        );
    }
    if want("twice") {
        let twice_sum = |(x, y): (&f32, &f32)| {
            let (x, y) = (bf16::from_f32(*x).to_f32(), bf16::from_f32(*y).to_f32());
            u32::from(bf16::from_f32(bf16::from_f32(x + y).to_f32() + y).to_bits())
        };
        variant!(
            ledger,
            ctx,
            "twice",
            bf16,
            [furiosa_opt_examples::runtime_rows_vector_silicon::wide],
            wide::twice,
            wide::twice_static,
            bf16::from_f32(-9876.0),
            |x: &f32| bf16::from_f32(*x),
            |x: &bf16| u32::from(x.to_bits()),
            twice_sum,
            counts(),
            iters
        );
    }
    if want("grouped") {
        variant!(
            ledger,
            ctx,
            "grouped",
            bf16,
            [furiosa_opt_examples::runtime_rows_vector_silicon::grouped],
            grouped::rows,
            grouped::rows_static,
            bf16::from_f32(-9876.0),
            |x: &f32| bf16::from_f32(*x),
            |x: &bf16| u32::from(x.to_bits()),
            bf16_sum,
            counts(),
            iters,
            8
        );
    }
    if want("tokenwise") {
        variant!(
            ledger,
            ctx,
            "tokenwise",
            bf16,
            [furiosa_opt_examples::runtime_rows_vector_silicon::tokenwise],
            tokenwise::rows,
            tokenwise::rows_static,
            bf16::from_f32(-9876.0),
            |x: &f32| bf16::from_f32(*x),
            |x: &bf16| u32::from(x.to_bits()),
            bf16_sum,
            vec![1, 8, 64, 127, 128, 256, 384, 512, 9, 0, -1, 513],
            iters,
            8
        );
    }
    for failure in &ledger.failures {
        println!("FAIL {failure}");
    }
    assert!(ledger.failures.is_empty(), "{} failing launches", ledger.failures.len());
    Ok(())
}
