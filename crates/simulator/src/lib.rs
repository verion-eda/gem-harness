// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Interactive RTL simulation against a gem-harness-compiled design.
//!
//! `simulator` is the public Rust API for the project. Given a
//! `.gemparts` file produced by [`compile`], it runs the design on
//! the GPU (with the `cuda` feature) or on a CPU fallback, exposes
//! step-by-step control over clocks and inputs, lets callers poke and
//! read arbitrary signals, supports DPI calls back into Rust, and writes
//! VCD traces.
//!
//! User-facing narrative documentation lives in the mdbook source under
//! `docs/`; this rustdoc is the developer reference for the API itself.
//!
//! # Pipeline context
//!
//! ```text
//!   gate-level Verilog
//!         │   (Yosys synthesis against aigpdk)
//!         ▼
//!   ┌─────────────┐     ┌─────────┐     ┌────────┐     ┌─────────┐
//!   │  aig parse  │ ──▶ │ staging │ ──▶ │ repcut │ ──▶ │ flatten │
//!   └─────────────┘     └─────────┘     └────────┘     └─────────┘
//!                                                            │
//!                                                            ▼
//!                                                      .gemparts file
//!                                                            │
//!                                              ┌─────────────┴────────────┐
//!                                              ▼                          ▼
//!                                           simulator                gem-simulator
//!                                       (interactive)                  (one-shot)
//! ```
//!
//! Start at [`SimulatorBuilder`] for the entry point.
//!
//! # Feature flags
//!
//! - **`cuda`** — GPU backend via [`gem-cuda`]. Requires the CUDA toolkit at
//!   build time. Without it, the CPU backend is used (correct, but orders of
//!   magnitude slower — useful for CI and local development).
//! - **`python`** — PyO3 bindings (the `py` module). Requires `cuda`.
//!
//! # Modules
//!
//! - [`sim`] — the [`InteractiveSimulator`] itself and its
//!   [`SimulatorBuilder`].
//! - [`signals`] — resolving hierarchical signal names against the design.
//! - [`offset`] — [`StateOffset`] handles for direct memory access.
//! - [`vcd`] — waveform tracing.
//! - `ffi` — C ABI for embedding in non-Rust testbenches *(cuda only)*.
//! - `py` — Python bindings *(cuda + python)*.
//!
//! [`compile`]: ../compile/
//! [`gem-cuda`]: ../gem_cuda/

pub(crate) mod cpu_simulator;
mod dpi;
pub mod offset;
#[cfg(feature = "cuda")]
mod pipe;
pub mod signals;
pub mod sim;
pub mod vcd;

#[cfg(feature = "cuda")]
pub mod ffi;

#[cfg(all(feature = "cuda", feature = "python"))]
pub mod py;

/// An AIG pin index (without invert bit).
pub type PinType = usize;

pub use dpi::DpiMode;
pub use offset::StateOffset;
pub use sim::{InteractiveSimulator, SimulatorBuilder, ToggleClock};
pub use vcd::{Tracer, WireHandle};
