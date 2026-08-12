# Tuning

Tuning is a controlled experiment: preserve correctness, record a baseline, change one supported lever, and compare the same schedule metric before deciding whether to keep the change.


## Control the experiment

Follow this order for every candidate:

1. Run the type-checking path to validate mapping and shape constraints.
2. Run emulation with a host oracle, including the relevant boundary cases.
3. Compile the baseline and record its schedule makespan.
4. Change one lever while keeping release, shapes, data types, and mappings fixed.
5. Run the same checks and dump the candidate schedule.
6. Keep the change only if correctness still passes and the measured metric improves; otherwise restore the baseline.

Schedule makespan is the primary static metric in this chapter. Do not report a throughput or cycle improvement without a reproducible schedule comparison or separately documented device evidence.

## Choose one lever

Choose a lever only after diagnosis names the dependency or resource it targets:

- change the execution-engine path when the current resource is the bottleneck;
- change a tile or split shape when the schedule exposes avoidable serial work or an ill-fitting reduction;
- change a mapping, padding, or transfer boundary when the limiting interval is movement or an address dependency.

The [Moving Tensors](../moving-tensors/index.md), [Computing Tensors](../computing-tensors/index.md), and [Mapping Tensors](../mapping-tensors/index.md) chapters define these choices. This page defines how to test them, not their contracts.

## Double buffering


The public `TrfAddress` API exposes `Full`, `FirstHalf`, and `SecondHalf`. A kernel can place one batch in one half while another operation uses the other half; the halves are capacity regions, not an unconditional promise of overlap. Both halves share banks, so a candidate must still be checked for resource contention in the schedule.

See [Register Files: Double Buffering](../computing-tensors/register-files.md#double-buffering) for capacities and address modes. Use the schedule viewer to verify whether the chosen pattern actually overlaps operations; do not assume a fixed cycle reduction.

## Decide and record

For each experiment, record the hypothesis, the changed lever, the baseline and candidate makespans, the correctness commands, and the decision. This record makes a later Qwen3 capstone reproducible without repeating the engine contracts here.
