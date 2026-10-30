//! The profile a function loads with, as `FURIOSA_OPT_PROFILE` and the tracing subscriber ask.

use std::sync::Once;

use furiosa_opt_rt::registry::{Level, Profile, TARGET};

/// The profile `FURIOSA_OPT_PROFILE` asks a function to load with, when a subscriber listens on
/// [`TARGET`]. Without one, a warning says so once per process.
pub(super) fn requested() -> Option<Profile> {
    static WARNING: Once = Once::new();
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
            refused: |level, error| eprintln!("furiosa-opt: profiling at {level:?} is refused: {error}"),
        });
    }
    WARNING.call_once(|| {
        eprintln!(
            "furiosa-opt: FURIOSA_OPT_PROFILE requested profiling, but no subscriber \
             listens on {}; install a tracing subscriber before loading functions; \
             profiling is disabled",
            TARGET,
        );
    });
    None
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

    /// Profiling requires a supported level and a listener; warning delivery needs no subscriber.
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
            for _ in 0..2 {
                assert_eq!(requested().map(|profile| profile.level), expected);
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
                    "backend::npu::profile::tests::profile_requires_listener",
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
