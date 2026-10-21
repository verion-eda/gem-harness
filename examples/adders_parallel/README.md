# Parallel Adders Example

Benchmark design for RTL simulator scaling: N parallel 32-bit synchronous adders.

## Usage

```bash
make synth          # N=32 (default)
make comp
make sim

# Or run full pipeline
make all
```

Override N:

```bash
make synth N=64
make comp
make sim
```

## Design

- `adder.sv` — N parallel 32-bit adders with registered outputs (1 cycle latency)
- `scripts/synthesize.tcl` — Yosys synthesis
- `adders_parallel_testbench.rs` — Rust testbench

Outputs: `build/gatelevel_N<N>.gv`, `build/result_N<N>.gemparts`

Compile and simulate require CUDA.
