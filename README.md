# gem-harness

`gem-harness` is an open-source RTL logic simulator with CUDA acceleration,
forked from [GEM](https://github.com/NVlabs/GEM), which is developed and
maintained by NVIDIA Research. See the
[paper](https://research.nvidia.com/publication/2025-06_gem-gpu-accelerated-emulator-inspired-rtl-simulation)
for a summary of the original work.

GEM and eda-infra-rs are not our work. gem-harness is published by Verion GmbH
as a modified version of both, under the same [Apache License 2.0](LICENSE).

> **This is a public archive** of our attempt to extend GEM into an interactive
> simulator. We do not maintain this code, and we do not plan to fix bugs or
> review changes here. It is published as-is, in the hope that someone finds it
> useful.

## Provenance

Every source file says where it comes from in its SPDX header:

- **From GEM** (Copyright NVIDIA): the `aigpdk/` cell library and most of the
  `aig`, `staging`, `repcut`, `flatten`, `gem-cuda`, `compile` and
  `gem-simulator` crates. These files carry the original NVIDIA copyright line;
  files we changed additionally carry a "Verion GmbH. Modifications to the
  original file." line. Files in these crates that carry only a Verion GmbH
  line are ours.
- **From [eda-infra-rs](https://github.com/gzz2000/eda-infra-rs)** (Copyright
  Zizheng Guo): the `netlistdb`, `ulib`, `ucc` and `vcd-ng` crates, vendored
  with modifications. `vcd-ng` is itself derived from Kevin Mehall's `vcd` crate
  and is MIT-licensed (see [crates/vcd-ng/LICENSE](crates/vcd-ng/LICENSE)).
  Files without a header in these crates are unmodified upstream files.
- **Our own code** (Copyright Verion GmbH): the `simulator` crate (interactive
  simulation, DPI, FFI and Python bindings), the examples, the Python package
  and the Nix setup.

We also depend on forks of
[sverilogparse](https://github.com/verion-eda/sverilogparse) (from
eda-infra-rs) and [mt-kahypar-rs](https://github.com/verion-eda/mt-kahypar-rs)
(Rust bindings to [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar)). See
[LICENSES.txt](LICENSES.txt) for all third-party notices.

## Examples

Most examples use [just](https://github.com/casey/just) (included in the Nix
shell). Run the full pipeline (synth → compile → simulate) with:

```bash
just examples/<name>/all
```

Available examples: `adder`, `adders_parallel`, `alu`, `chained-adders`,
`const-dff`, `traffic_light`, `python`

The individual steps are `synth`, `comp` and `sim`. The `dpi` and `ffi` examples
use Makefiles instead: `make -C examples/<name>`. See
[examples/readme.md](examples/readme.md) for details.

## Development Setup

### Nix

gem-harness provides a Nix flake for reproducible development environments that
includes Rust toolchain, Yosys, and other development tools.

**Prerequisites:**

- Install [Nix](https://nixos.org/download.html) with flakes support
- Enable experimental features by creating `~/.config/nix/nix.conf`:
  ```
  experimental-features = nix-command flakes
  ```

**Quick Start:**

```bash
# Enter the development environment
nix develop
```

The Nix environment includes:

- Rust toolchain (automatically detected from `rust-toolchain.toml` if present)
- Nightly rustfmt for code formatting
- Yosys (EDA synthesis tool)
- cargo-deny, cargo-edit, cargo-watch
- rust-analyzer for IDE support
- pkg-config

**Note:** CUDA support requires system-level CUDA toolkit installation. Install
CUDA from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
separately, and the build system will detect it automatically.

## Compile and Run Your Design with GEM

gem-harness works in a way similar to an FPGA-based RTL emulator. It first
synthesizes your design with a special and-inverter graph (AIG) process, and
then map the synthesized gate-level netlist to a virtual manycore Boolean
processor which can be emulated with CUDA-compatible GPUs.

The synthesis and mapping is slower than the compiling/elaboration process of
CPU-based simulators. But it is a one-time cost and your design can be simulated
under different testbenches without re-running the synthesis or mapping.

## FFI Bindings

### Build and Install

```bash
cargo build --release --features "cuda" -p simulator
```

The C header can be generate using cbindgen:

```bash
cbindgen --crate simulator --output include/harness.h
```

### Example

See `examples/ffi/` and compile/run with:

```bash
make -C examples/ffi
```

## Python Bindings

This document describes how to build and use the Python bindings for the
gem-harness simulator.

The Python bindings provide a Pythonic interface to the simulator using PyO3.
These bindings offer:

- **Type safety**: Full type hints with `.pyi` stub files
- **Pythonic API**: Context managers, exceptions, and idiomatic Python patterns
- **Performance**: Direct Rust calls without C FFI overhead
- **Ergonomics**: Both pin-based and signal-name-based APIs

Note that the type stubs are currently manually generated! If we change things
in the bindings we need to update the stubs.

### Prerequisites

1. Rust toolchain (cargo, rustc)
2. Python 3.10 or later
3. CUDA toolkit (for GPU acceleration)
4. uv (fast Python package manager)

### Build and Install

```bash
uv sync
uv run maturin build --features python,cuda
# note maturin develop will fail on nixos but leave the shared library in the python/harness folder
uv run maturin develop --features python,cuda
uv run examples/python/example.py
```

### Example

See `examples/python/` and run with:

```bash
uv run examples/python/example.py
```

### Type Hints

The module includes a `harness.pyi` stub file for full type checking support
with mypy, pyright, and other type checkers:

```python
from harness import Simulator

# type checkers will validate this
sim: Simulator = Simulator("netlist.v", "gemparts.txt", 256, 0)
result: int = sim.peek_signal("output")
```

## DPI Support

gem-harness supports `import "DPI-C"` functions and DPI with Rust closures,
allowing host-side callbacks to be invoked on every simulation cycle. This is
useful for modelling components that cannot be expressed as synthesizable logic
(e.g., complex arithmetic, external memory models, cosimulation bridges).

See the [DPI example](examples/dpi/) for a complete working example.
