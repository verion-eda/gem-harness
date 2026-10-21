# Python Bindings for gem-harness

This document describes how to build and use the Python bindings for the
hardware simulator.

## Overview

The Python bindings provide a Pythonic interface to the simulator using PyO3.
These bindings offer:

- **Type safety**: Full type hints with `.pyi` stub files
- **Pythonic API**: Context managers, exceptions, and idiomatic Python patterns
- **Performance**: Direct Rust calls without C FFI overhead
- **Ergonomics**: Both pin-based and signal-name-based APIs

Note that the type stubs are currently manually generated! If we change things
in the bindings we need to update the stubs.

## Building

### Prerequisites

1. Rust toolchain (cargo, rustc)
2. Python 3.10 or later
3. CUDA toolkit (for GPU acceleration)
4. uv (fast Python package manager)

### Build and Install

```bash
# Create a virtual environment
uv sync
uv run maturin build --features python,cuda
# note this will fail on nixos but leave the so library in the python/harness folder
# only works like this
uv run maturin develop --features python,cuda
uv run examples/python/example.py
```

## Usage

### Basic Example

```python
from harness import Simulator

# Create simulator
sim = Simulator(
    netlist_path="netlist.v",
    gemparts_path="gemparts.txt",
    num_blocks=1,
    device_id=0
)

# Reset and run simulation
sim.reset()
sim.poke_signal("clk", True)
sim.poke_signal("data[7:0]", 0x42)
sim.step()
result = sim.peek_signal("output[7:0]")
print(f"Output: 0x{result:02x}")
```

### Context Manager

The simulator supports Python's context manager protocol for automatic cleanup:

```python
with Simulator("netlist.v", "gemparts.txt", 256, 0) as sim:
    sim.reset()
    sim.poke_signal("input", True)
    sim.step()
    output = sim.peek_signal("output")
# sim.sync() is called automatically on exit
```

### API Methods

#### `__init__(netlist_path, gemparts_path, num_blocks, device_id)`

Create a new simulator instance.

#### `reset()`

Reset the simulator to initial state. Must be called before any poke/step/peek
operations.

#### `poke(aigpin: int, value: bool)`

Set an input signal by AIG pin number.

#### `poke_signal(signal_path: str, value: bool)`

Set an input signal by name (e.g., `"clk"`, `"cpu/data[7:0]"`).

#### `poke_bits(aigpins: list[int], value: int)`

Set multiple input signals at once. More efficient than multiple `poke()` calls.

#### `step()`

Execute one simulation cycle.

#### `peek(aigpin: int) -> bool`

Read an output signal by AIG pin number.

#### `peek_signal(signal_path: str) -> int`

Read an output signal by name. Returns a 64-bit unsigned integer.

#### `peek_bits(aigpins: list[int]) -> int`

Read multiple output signals at once. More efficient than multiple `peek()`
calls.

#### `sync()`

Synchronize GPU operations (wait for completion).

#### `num_inputs() -> int`

Get the number of input pins in the circuit.

#### `num_outputs() -> int`

Get the number of output pins in the circuit.

## Type Hints

The module includes a `harness.pyi` stub file for full type checking support with
mypy, pyright, and other type checkers:

```python
from harness import Simulator

# Type checkers will validate this
sim: Simulator = Simulator("netlist.v", "gemparts.txt", 256, 0)
result: int = sim.peek_signal("output")
```

## Multi-bit Operations

For better performance when working with buses, use the `_bits` variants:

```python
# Instead of:
for i in range(8):
    sim.poke(base_pin + i, (value >> i) & 1)

# Use:
pins = list(range(base_pin, base_pin + 8))
sim.poke_bits(pins, value)
```

## Examples

See `examples/python/example.py` for a comprehensive example demonstrating:

- Signal-based API
- Pin-based API
- Multi-bit operations
- Context managers
- Multi-cycle simulation

## Integration with Python Ecosystem

The bindings work seamlessly with Python's ecosystem:

```python
import numpy as np
from harness import Simulator

sim = Simulator(...)
sim.reset()

# Use NumPy arrays for test vectors
test_vectors = np.array([0x00, 0xFF, 0x42, 0xAA], dtype=np.uint8)

results = []
for value in test_vectors:
    sim.poke_bits([10, 11, 12, 13, 14, 15, 16, 17], int(value))
    sim.step()
    results.append(sim.peek_signal("output"))

results = np.array(results, dtype=np.uint64)
```

## Performance Tips

1. **Use `_bits` methods** for multi-bit signals (more efficient than individual
   pokes/peeks)
2. **Batch operations** when possible instead of sync'ing after every cycle
3. **Use signal names** when available (more maintainable than pin numbers)
4. **Call `sync()`** only when you need to ensure GPU completion (e.g., for
   timing)

## License

Same as the main gem-harness project (Apache-2.0).
