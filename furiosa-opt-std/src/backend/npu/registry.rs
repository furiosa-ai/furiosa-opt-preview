//! The native registry's functions, loaded once per device.

use std::sync::OnceLock;

use furiosa_opt_rt as rt;
use rt::registry::Key;

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
pub(super) fn index(path: &'static str, key: Key<'_>) -> Result<usize, Error> {
    rt::registry::find(path, key).ok_or_else(|| match key {
        Key::Numeric(key) => Error::Uncompiled {
            path,
            key: key.to_vec(),
        },
        Key::Type(instance) => Error::UncompiledInstance { path, instance },
    })
}

/// Registry entry `index` on `device`, loaded on first use.
pub(super) fn function(device: &<Npu as Runtime>::Device, index: usize) -> Result<&rt::Function, Error> {
    device.registry.functions[index].get_or_try_init(|| {
        // SAFETY: the link pass wrote every entry with the compiler's image of its function.
        let load = unsafe { rt::registry::entries()[index].load(&device.device, super::profile::requested()) };
        // Cold loading must progress even when the caller's Tokio budget is exhausted.
        Ok(futures_executor::block_on(tokio::task::unconstrained(load))?)
    })
}
