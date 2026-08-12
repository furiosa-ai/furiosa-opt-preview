# Scheduling and Tuning

Scheduling and tuning turn a correct kernel into a measured execution plan: inspect the schedule, identify the limiting dependency or resource, change one supported choice, and compare the result under the same conditions.


Use this chapter after [Computing Tensors](../computing-tensors/index.md) when context overlap, memory dependencies, or resource conflicts affect the schedule. The chapter follows one loop:

1. [Schedule](./schedule.md): understand contexts, dependencies, and operation order.
2. [Diagnosis](./diagnosis.md): inspect a dumped schedule and state a bottleneck hypothesis.
3. [Tuning](./tuning.md): test one change, compare the same metric, and keep or revert it.

The chapter does not re-explain mapping, tensor movement, or compute-engine contracts. It links to those chapters when a scheduling decision depends on them.

## Evidence boundary

Treat each result as a separate kind of evidence:

- type checking establishes mapping and shape validity;
- emulation with a host oracle establishes values;
- a dumped schedule establishes static execution order, overlap, and makespan;
- an NPU run, when available, establishes device behavior under its stated release and inputs.

A passing emulation test is not a performance result. Compare schedule makespan only when release, shapes, data types, and mappings are unchanged.
