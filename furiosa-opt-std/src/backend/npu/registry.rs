//! The native registry's functions, loaded once per device.

use std::sync::OnceLock;

use furiosa_opt_rt as rt;

use super::Npu;
use crate::Error;
use crate::backend::Runtime;

pub(super) struct Registry {
    functions: Box<[OnceLock<rt::Function>]>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            functions: rt::registry::entries().iter().map(|_| OnceLock::new()).collect(),
        }
    }
}

/// Where registry entry `path`/`key` sits.
pub(super) fn index(path: &'static str, key: &[usize]) -> Result<usize, Error> {
    rt::registry::find(path, key).ok_or_else(|| Error::Uncompiled {
        path,
        key: key.to_vec(),
    })
}

/// Registry entry `index` on `device`, loaded on first use.
pub(super) fn function(device: &<Npu as Runtime>::Device, index: usize) -> Result<&rt::Function, Error> {
    device.registry.functions[index].get_or_try_init(|| {
        // SAFETY: the link pass wrote every entry with the compiler's image of its function.
        let load = unsafe {
            rt::registry::load(
                &rt::registry::entries()[index],
                &device.device,
                super::profile::requested(),
            )
        };
        Ok(wait(load)?)
    })
}

/// Runs `future` to its end on this thread, parking between polls. Unlike an executor's
/// `block_on`, it enters no executor and consumes no task budget, so it may run inside a caller's
/// executor; the device worker thread progresses a load, so parking here cannot stall it.
fn wait<F: Future>(future: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Unpark(std::thread::current())));
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut std::task::Context::from_waker(&waker)) {
            return output;
        }
        std::thread::park();
    }
}
