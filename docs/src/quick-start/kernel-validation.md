# Kernel Validation

The validation sequence distinguishes structural validity, numerical correctness, hardware behavior, and static schedule data.

1. **Typecheck** executes the kernel body with phantom tensors to catch mapping and shape errors.
2. **Emulation** compares host-side tensor results with a host oracle.
3. **NPU execution** runs the compiled executable device format (EDF) through the Furiosa SDK and physical hardware.
4. **Schedule data** inspects emitted schedule JSON and compares makespan with release, shapes, types, and mappings fixed.

Typecheck executes the kernel body with phantom tensors, so an elementwise comparison or assertion loop can execute zero iterations and still pass.
That means typecheck can catch mapping and shape errors without providing numerical-correctness proof.
Compare schedule makespan only when the release, shapes, data types, and mapping are fixed.
A schedule is not a throughput measurement by itself.
Use [Scheduling and Tuning](../scheduling/index.md) to compare fixed schedule candidates.
Record the source revision, backend, shapes, mappings, oracle result, and schedule artifact for each comparison.
