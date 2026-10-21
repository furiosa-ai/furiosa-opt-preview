# furiosa-opt-abi

What the host side (compiler, `furiosa-opt-rt`) and the device side (firmware, bootloader) of an RNGD device agree on.
`no_std`, so the device side builds on it.
Applications use `furiosa-opt-std` instead.

| module | contract |
|---|---|
| `executable` | the compiled device code the firmware loads and runs |
| `bootloader` | the request that boots the firmware, and its answer |
| `ring`, `doorbell` | the queues that carry messages between host and firmware |
| `memory`, `Topology` | device memory kinds and device shapes |
| `mmio` | register access, and the barriers between register and memory accesses |
| `arch::rngd` | the RNGD hardware's addresses and limits |
