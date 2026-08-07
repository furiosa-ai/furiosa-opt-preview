---
name: furiosa-opt-kernel-authoring
description: Build, debug, or optimize Furiosa Optimizer TCP kernels with a defined mathematical contract, verified pattern selection, typecheck and emulation evidence, and Schedule Viewer makespan review. Use when implementing or reviewing a `furiosa-opt` kernel, mapping, Tensor Unit pipeline, or schedule-performance change.
---

# Furiosa Optimizer Kernel Authoring

Follow `docs/src/quick-start.md` for the kernel quality model and verified patterns.

## Agent Procedure

1. Record the release tag and commit, or the checked-out contributor commit.
2. Implement the selected pattern, host oracle, and adversarial cases together.
3. Run the evidence sequence:

   ```bash
   cargo furiosa-opt --backend typecheck run --release --bin <name>
   cargo furiosa-opt --backend emulation test --release --bin <name>
   cargo furiosa-opt compile <device-function> --dump-schedule schedule.json
   ```

   Run an NPU test when hardware is available.
4. Return the selected pattern, changed files, command results, schedule comparison, remaining hardware gap, and novel dataflow choices.

## Constraints

- Treat `emulation`, `typecheck`, and `npu` as the supported backends. `emulation` is the default.
- Treat a passing typecheck as mapping evidence, not value evidence.
- Treat a passing emulation test as correctness evidence, not schedule evidence.
- Do not claim a performance improvement without a lower makespan against the recorded baseline.
- Do not update expected values, tests, or schedules merely to make a failure disappear. Diagnose the causal mapping, dataflow, or schedule decision.
