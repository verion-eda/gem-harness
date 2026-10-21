# Chained Adders Example

This example demonstrates a design with multiple stages of computation to help debug multi-stage compilation (vertical splits of the DAG) in GEM.

## Purpose

The OpenPiton FPU example is the first example that requires multiple stages in the compilation due to the complexity and depth of the circuit. This simpler example provides an easier-to-debug testbed for multi-stage compilation issues.

## Design Structure

The `chained_adders` module creates a multi-stage adder pipeline:

1. **Stage 1 (Parallel)**: Two independent adders compute `sum_ab = a + b` and `sum_cd = c + d`
2. **Stage 2 (Sequential)**: Combines stage 1 results: `sum_total = sum_ab + sum_cd`
3. **Stage 3 (Complex chain)**: Further computation: `sum_complex = sum_total + a + d`

This structure creates clear vertical splits in the DAG, making it ideal for testing and debugging:

- Parallel operations at stage 1
- Dependencies between stages (stage 2 depends on stage 1)
- Additional depth for more complex multi-stage scenarios

## Building and Running

**Note**: The compile and simulate steps require CUDA. If CUDA is not available, you can still synthesize the design to generate the gate-level netlist.

```bash
# From the examples/chained-adders directory:

# Synthesize the design (generates gatelevel.gv)
make synthesize

# Compile to GEM format (requires CUDA)
make compile

# Run the testbench (requires CUDA)
make simulate

# Or run all steps (requires CUDA)
make all

# Clean generated files
make clean
```

### Without CUDA

If you just want to inspect the synthesized gate-level netlist without running the full simulation:

```bash
make synthesize
# Inspect the generated gatelevel.gv file
```

## Files

- `chained_adders.sv` - SystemVerilog design with multi-stage adder chain
- `chained_adders_testbench.rs` - Rust testbench using GEM simulator
- `synthesise.tcl` - Yosys synthesis script
- `Makefile` - Build automation
- `gatelevel.gv` - Generated gate-level netlist (after synthesis)
- `result.gemparts` - Generated GEM compiled format (after compile)

## Testing

The testbench runs several test cases with different input combinations and verifies all output stages. It checks:

- Intermediate results (sum_ab, sum_cd)
- Final combined results (sum_total, sum_complex)

Success is indicated by all tests passing with a ✓ marker.
