#[expect(dead_code, missing_docs, reason = "bindgen output; bindgen does not emit docs")]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::NpuDesc;
pub(crate) use bindings::*;

use std::sync::OnceLock;

use crate::context::Device;

pub(crate) fn rt() -> *const Runtime {
    struct Handle(*mut Runtime);
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    static RT: OnceLock<Handle> = OnceLock::new();
    RT.get_or_init(|| {
        let ver = unsafe { std::ffi::CStr::from_ptr(furiosa_version()) };
        log::info!("device-runtime: {}", ver.to_string_lossy());
        let Device { chip: chips, pe: pes } = *DEVICE.get_or_init(|| Device { chip: 1, pe: 8 });
        let npus: Vec<NpuDesc> = (0..chips)
            .flat_map(|chip| match pes {
                8 => vec![
                    NpuDesc {
                        chip,
                        pe_start: 0,
                        pe_end: 3,
                    },
                    NpuDesc {
                        chip,
                        pe_start: 4,
                        pe_end: 7,
                    },
                ],
                n => vec![NpuDesc {
                    chip,
                    pe_start: 0,
                    pe_end: n - 1,
                }],
            })
            .collect();
        let ptr = unsafe { furiosa_runtime_init(npus.as_ptr(), npus.len()) };
        assert!(!ptr.is_null(), "failed to acquire NPU");
        Handle(ptr)
    })
    .0
}

static DEVICE: OnceLock<Device> = OnceLock::new();

/// Binds the `#[device(chip, pe)]` the process runs on, via `Context::acquire().bind(..)`.
/// First bind wins; a conflicting one panics.
pub(crate) fn bind_device(device: Device) {
    let cur = *DEVICE.get_or_init(|| device);
    assert_eq!(
        cur, device,
        "conflicting NPU device in one process: {cur:?} vs {device:?}"
    );
}
