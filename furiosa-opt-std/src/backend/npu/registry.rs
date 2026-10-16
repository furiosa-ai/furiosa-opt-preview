//! Native compiler programs linked into this executable.

use std::sync::{Arc, Once, OnceLock};

use furiosa_opt_rt as rt;

use super::Npu;
use crate::Error;
use crate::backend::Runtime;

use rt::registry::{Entry, Key, Level, Profile, TARGET};

pub(super) struct Registry {
    functions: Box<[OnceLock<rt::Function>]>,
    warning: Once,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            functions: entries().iter().map(|_| OnceLock::new()).collect(),
            warning: Once::new(),
        }
    }
}

impl Registry {
    fn profile(&self) -> Option<Profile> {
        let spec = std::env::var_os("FURIOSA_OPT_PROFILE")?;
        let level = match spec.to_str().and_then(|spec| spec.parse().ok()) {
            Some(tracing::Level::INFO) => Level::Info,
            Some(tracing::Level::DEBUG) => Level::Debug,
            Some(tracing::Level::TRACE) => Level::Trace,
            _ => {
                eprintln!(
                    "furiosa-opt: invalid FURIOSA_OPT_PROFILE={spec:?}; \
                     expected info, debug, or trace; profiling is disabled"
                );
                return None;
            }
        };
        if tracing::span_enabled!(target: TARGET, tracing::Level::INFO) {
            return Some(Profile {
                level,
                report: |level, error| eprintln!("furiosa-opt: profiling at {level:?} is refused: {error}"),
            });
        }
        self.warning.call_once(|| {
            eprintln!(
                "furiosa-opt: FURIOSA_OPT_PROFILE requested profiling, but no subscriber \
                 listens on {}; install a tracing subscriber before loading functions; \
                 profiling is disabled",
                TARGET,
            );
        });
        None
    }
}

// One contract with the driver's link pass (npu-opt `link.rs`): section `furiosa_kernels` holds
// `rt::registry::Entry` values sorted by path then key, between these linker-defined bounds.
unsafe extern "C" {
    #[link_name = "__start_furiosa_kernels"]
    static START: u8;
    #[link_name = "__stop_furiosa_kernels"]
    static STOP: u8;
}

// Defines the section and its bounds in a link the driver never saw (a plain cargo build), so
// that binary's registry is empty instead of failing to link.
#[used]
#[unsafe(link_section = "furiosa_kernels")]
static ANCHOR: [Entry; 0] = [];

/// The registry the link pass emits, sorted by path and key.
pub(super) fn entries() -> &'static [Entry] {
    std::hint::black_box(&raw const ANCHOR);
    let start = &raw const START;
    let stop = &raw const STOP;
    let bytes = stop.addr() - start.addr();
    assert!(
        bytes.is_multiple_of(std::mem::size_of::<Entry>()),
        "malformed native registry"
    );
    // SAFETY: the bounds delimit the section's `Entry` values, which the link pass emitted with the
    // same rustc and layout as this crate (`rt::registry::Entry`).
    unsafe { std::slice::from_raw_parts(start.cast::<Entry>(), bytes / std::mem::size_of::<Entry>()) }
}

/// Where registry entry `path`/`key` sits. The link pass assembles the registry after every crate
/// is compiled, so a function knows its unique `(path, key)` but not where its entry landed.
pub(super) fn index(path: &'static str, key: Key<'_>) -> Result<usize, Error> {
    entries()
        .binary_search_by(|entry| {
            let order = entry.key().cmp(&key);
            entry.path.cmp(path).then(order)
        })
        .map_err(|_| match key {
            Key::Numeric(key) => Error::Uncompiled {
                path,
                key: key.to_vec(),
            },
            Key::Type(instance) => Error::UncompiledInstance { path, instance },
        })
}

/// Registry entry `index` on `device`, loaded by the compiler's loader on first use.
pub(super) fn function(device: &<Npu as Runtime>::Device, index: usize) -> Result<&rt::Function, Error> {
    // Cold loading must progress even when the caller's Tokio budget is exhausted.
    device.registry.functions[index].get_or_try_init(|| {
        let load = (entries()[index].image)(Arc::clone(&device.device), device.registry.profile());
        Ok(futures_executor::block_on(tokio::task::unconstrained(load))?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Listener;

    impl tracing::Subscriber for Listener {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            metadata.target() == TARGET
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, _: &tracing::Event<'_>) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    /// Capture requires a supported level and a listener; warning delivery needs no subscriber.
    #[test]
    fn profile_requires_listener() {
        if let Ok(case) = std::env::var("FURIOSA_OPT_PROFILE_TEST") {
            let expected = match case.as_str() {
                "info" => Some(Level::Info),
                "debug" => Some(Level::Debug),
                "trace" => Some(Level::Trace),
                _ => None,
            };
            if expected.is_some() {
                tracing::subscriber::set_global_default(Listener).unwrap();
            }
            let registry = Registry::default();
            for _ in 0..2 {
                assert_eq!(registry.profile().map(|profile| profile.level), expected);
            }
            return;
        }
        for (case, value) in [
            ("unset", None),
            ("invalid", Some("typo")),
            ("missing", Some("trace")),
            ("info", Some("info")),
            ("debug", Some("debug")),
            ("trace", Some("trace")),
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "backend::npu::registry::tests::profile_requires_listener",
                    "--nocapture",
                ])
                .env("FURIOSA_OPT_PROFILE_TEST", case)
                .env_remove("FURIOSA_OPT_PROFILE")
                .envs(value.map(|value| ("FURIOSA_OPT_PROFILE", value)))
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{case}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(!output.stderr.is_empty(), matches!(case, "invalid" | "missing"));
        }
    }
}
