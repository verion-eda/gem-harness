# DPI Examples

Demonstrates `import "DPI-C"` support in the gem-harness simulator (CPU and
GPU).

DPI (Direct Programming Interface) lets a hardware design call external software
functions during simulation. This is used when part of the system (e.g. a memory
model or debug interface) is easier to write in C/Rust than in hardware
description language. The design declares a function signature, and at
simulation time a callback provides the implementation.

Five examples are included, each testing a different aspect of DPI support:

| Example            | What it tests                                                                                                                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dpi_counter`      | Basic DPI: a function that takes two 32-bit inputs, returns one 32-bit output (saturating add). The simplest case — one call, scalar types, return value.                                          |
| `dpi_output_param` | A function that writes results via output parameters instead of a return value. The preprocessor must convert the output variable into a cell-driven net and remove any conflicting assignments.   |
| `dpi_alltypes`     | Exercises all supported scalar types (`byte`, `shortint`, `int`, `longint`, `chandle`, `string`, `real`, `shortreal`) and functions with multiple arguments, to verify correct bit-width handling. |
| `dpi_const_input`  | DPI inputs wired to constant parameters (e.g. `CHIP_ID=0`). These get constant-folded during synthesis, so the simulator must recognize them as fixed values rather than state-vector reads.       |
| `dpi_array`        | Unpacked `byte[]` array parameters (preprocessor test only, no simulation testbench). Verifies that the preprocessor correctly converts unpacked arrays to packed bit vectors before synthesis.    |

## Build & Run

### DPI Counter (32-bit)

```
cd examples/dpi
make              # preprocess → synthesize → compile → simulate (dpi_counter)
make output_param # same pipeline for dpi_output_param
make alltypes     # same pipeline for dpi_alltypes (all DPI types + multi-arg)
make const_input  # same pipeline for dpi_const_input
make test_array   # run preprocessor tests for unpacked byte[] arrays
```

Individual steps: `make preprocess`, `make synth`, `make comp`, `make sim`.

The counter testbench supports a `--dpi-lib` flag to load callbacks from a C
shared library instead of Rust closures:

```
make sim_c      # CPU, C shared library
make sim_c_gpu  # GPU, C shared library (requires --features cuda)
```

## How It Works

1. `scripts/dpi_preprocess.py`
   ([pyslang](https://github.com/MikePopoloski/slang)) rewrites `import "DPI-C"`
   calls into `(* blackbox *)` modules named `$__DPI__<func>`.
2. Yosys synthesizes the preprocessed design; the gem-harness compiler preserves
   DPI blackboxes in the AIG.
3. At simulation time, callbacks are registered via `SimulatorBuilder`:

There are two ways to register DPI callbacks:

**Rust closures** — implement the function directly in Rust:

```rust
SimulatorBuilder::new(gatelevel, gemparts, num_blocks)
    .cpu()
    .register_dpi("sat_add", |args| {
        let a = args[0] as i32;
        let b = args[1] as i32;
        vec![a.saturating_add(b) as u32]
    })
    .build()?
```

Each port occupies `ceil(width / 32)` little-endian `u32` words in both `args`
and the return value.

Callbacks fire **before** each simulation cycle: inputs are read from the state
vector, outputs are written back, then the circuit evaluates.

## Limitations

- Requires `pyslang` (`pip install pyslang`); the Makefile expects python to be
  provided via `uv` (override with `PYTHON=...`).
- Only (C calling SV) is supported. `export "DPI-C"` (SV called from C) is not
  supported.
- GPU DPI adds per-cycle kernel launch overhead vs. batched non-DPI runs.
