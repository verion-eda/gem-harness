# Examples

This crate contains examples. We use/support two build types:

- Makefiles: for more involved builds
- Justfiles: for simpler examples sharing a common build pattern

## Simple Justfile Examples

- adder: simple adder
- adders_parallel: benchmark larger setting with variable amount of adders
- alu: simple alu
- chained-adders: larger adder by chaining
- const-dff: simple "unit"-test for constant dffs (had issues with those)
- traffic_light: medium sized traffic light circuit

The `python` example showcases the Python bindings (simple adder) also falls
into this category.

## Complexer Makefile Examples

- dpi: "unit"-tests and benchmarks for DPI

The `ffi` example showcases the FFI bindings (simple adder).
