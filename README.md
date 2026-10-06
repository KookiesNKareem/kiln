# accel-sim

Research code for designing AI accelerators with LLM-driven search.

- `kiln/`: **kiln**, a Rust simulator for accelerator systems: hardware and workload IR, cost model, mapper, Tier A engine, physical model (area, power, placement, wires), calibration, visualizer, Python API, and the `kiln_evo` design-search loop.
- `spec/`: the kiln spec (start with `spec/README.md`; decisions in `spec/08-decisions.md`).
- `calibration/`: benchmark runners and measured data (A100-40GB, TPU v5e, TPU v6e).
- `harness/`, `hw/`, `workloads/`, `scripts/`: the earlier Stream-based harness, kept as a reference.

```sh
cd kiln
cargo test --workspace --release
cargo run --release -p kiln-cli -- eval designs/reference/a100_sxm4_40gb.json5 --workload llama3_8b:decode_b1
```

Status: early and unvalidated beyond the measurements in `calibration/`. Not ready for outside use.
