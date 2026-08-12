# Diagnosis

Diagnosis converts a schedule view into one testable bottleneck hypothesis; it does not establish a speedup by inspection alone.

## Dump a schedule

Compile one device function with `--dump-schedule` and open the resulting JSON in the [Schedule Viewer](../appendix/schedule-viewer.md):

```bash
cargo furiosa-opt compile <device-function> \
  --dump-schedule schedule.json
furiosa-schedule-viewer
```

The viewer exposes the scheduled timeline, operation context, lifetime, connected tensors, and operator description. Select a node to trace its inputs and outputs; zoom the cycle range to isolate the suspected region. The appendix documents installation, filtering, and the viewer controls.

## Read the timeline

Start with the final scheduled cycle (the makespan), then inspect the intervals that determine it:

1. Identify the interval that reaches the final cycle.
2. Check whether its predecessor is a memory dependency, a shared resource, or an unscheduled gap.
3. Compare the relevant context lanes to see whether another operation could overlap.
4. State one candidate change and the reason it should alter that interval.

The schedule shows static compiler behavior. It does not replace a value oracle, and it does not prove device throughput.

## Keep the comparison valid

Record the source revision, release profile, device-function filter, shapes, element types, mappings, and schedule file for each candidate. Change only the selected kernel choice between candidates. If any of those inputs changes, treat the result as a new baseline rather than a tuning comparison.
