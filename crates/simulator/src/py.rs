// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Python bindings for InteractiveSimulator using PyO3
//!
//! This module provides Pythonic bindings that wrap the Rust
//! InteractiveSimulator directly, bypassing the C FFI layer for better type
//! safety and ergonomics.

#![allow(non_local_definitions)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::path::PathBuf;

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use ulib::Device;

use crate::{
    signals::{SignalError, SignalPath},
    sim::{InteractiveSimulator, SimulatorBuilder},
};

impl From<SignalError> for PyErr {
    fn from(error: SignalError) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }
}

/// Interactive hardware simulator for AIG netlists
///
/// This class provides a Python interface to simulate digital circuits
/// represented as And-Inverter Graphs (AIGs). It runs on CUDA GPUs for
/// high-performance simulation.
///
/// ## Arguments
///
/// * `netlist_path` - Path to the Verilog netlist file
/// * `gemparts_path` - Path to the GEM partitions file
/// * `num_blocks` - Number of CUDA blocks to use for simulation
/// * `device_id` - CUDA device ID (0 for first GPU)
///
/// ## Example
///
/// ```
/// sim = PySimulator("netlist.v", "gemparts.txt", num_blocks=1, device_id=0)
/// sim.reset()
/// sim.poke_signal("clk", True)
/// sim.step()
/// result = sim.peek_signal("output")
/// ```
#[pyclass(name = "Simulator")]
pub struct PySimulator {
    sim: InteractiveSimulator,
    initialized: bool,
}

#[pymethods]
impl PySimulator {
    #[new]
    fn new(
        netlist_path: String,
        gemparts_path: String,
        num_blocks: usize,
        device_id: u8,
    ) -> PyResult<Self> {
        let sim = SimulatorBuilder::new(
            PathBuf::from(netlist_path),
            PathBuf::from(gemparts_path),
            num_blocks,
        )
        .device(Device::CUDA(device_id))
        .build()
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to build simulator: {}", e))
        })?;

        Ok(PySimulator {
            sim,
            initialized: false,
        })
    }

    /// Reset the simulator to initial state
    ///
    /// This must be called once after creating the simulator and before
    /// any poke/step/peek operations.
    ///
    /// ## Example
    ///
    /// ```
    /// sim.reset()
    /// ```
    fn reset(&mut self) {
        self.sim.reset();
        self.initialized = true;
    }

    /// Set an input signal value by AIG pin number
    ///
    /// ## Arguments
    ///
    /// * `aigpin` - AIG pin number (without invert bit)
    /// * `value` - Signal value (True for high, False for low)
    ///
    /// ## Raises
    ///
    /// * `RuntimeError` - If reset() hasn't been called yet
    ///
    /// ## Example
    ///
    /// ```
    /// sim.poke(42, True)
    /// ```
    fn poke(&mut self, aigpin: usize, value: bool) -> PyResult<()> {
        if !self.initialized {
            return Err(PyRuntimeError::new_err("Must call reset() first"));
        }
        self.sim.poke(aigpin, value);
        Ok(())
    }

    /// Set an input signal value by signal name
    ///
    /// ## Arguments
    ///
    /// * `signal_path` - Signal name (e.g., "clk" or "cpu/alu/in[0]")
    /// * `value` - Signal value (True for high, False for low)
    ///
    /// ## Raises
    ///
    /// * `RuntimeError` - If reset() hasn't been called or signal not found
    ///
    /// ## Example
    ///
    /// ```
    /// sim.poke_signal("cpu/clk", True)
    /// sim.poke_signal("data[7:0]", False)
    /// ```
    fn poke_signal(&mut self, signal_path: &str, value: bool) -> PyResult<()> {
        if !self.initialized {
            return Err(PyRuntimeError::new_err("Must call reset() first"));
        }
        let signal_path: SignalPath = signal_path.parse()?;
        self.sim.poke_signal(&signal_path, value)?;
        Ok(())
    }

    /// Set multiple input signal values at once
    ///
    /// This is more efficient than calling poke() multiple times.
    /// The value is interpreted as a bit vector where bit i corresponds
    /// to aigpins[i].
    ///
    /// ## Arguments
    ///
    /// * `aigpins` - List of AIG pin numbers (without invert bit)
    /// * `value` - 64-bit value where bit i corresponds to aigpins[i]
    ///
    /// ## Raises
    ///
    /// * `RuntimeError` - If reset() hasn't been called or pins not found
    /// * `ValueError` - If more than 64 pins are provided
    ///
    /// ## Example
    ///
    /// ```
    /// # Set pins 10,11,12,13 to binary 1010
    /// sim.poke_bits([10, 11, 12, 13], 0b1010)
    /// ```
    fn poke_bits(&mut self, aigpins: Vec<usize>, value: u64) -> PyResult<()> {
        if !self.initialized {
            return Err(PyRuntimeError::new_err("Must call reset() first"));
        }
        if aigpins.len() > 64 {
            return Err(PyValueError::new_err(
                "Cannot poke more than 64 bits at once",
            ));
        }
        self.sim.poke_bits(&aigpins, value)?;
        Ok(())
    }

    /// Execute one simulation cycle
    ///
    /// This propagates the input values through the circuit logic.
    /// Call poke/poke_signal before step, then peek/peek_signal after.
    ///
    /// ## Example
    ///
    /// ```
    /// sim.poke_signal("clk", True)
    /// sim.step()
    /// result = sim.peek_signal("output")
    /// ```
    fn step(&mut self) {
        self.sim.step();
    }

    /// Read an output signal value by AIG pin number
    ///
    /// ## Arguments
    ///
    /// * `aigpin` - AIG pin number (with invert bit in LSB)
    ///
    /// ## Returns
    ///
    /// Signal value (True for high, False for low)
    ///
    /// ## Example
    ///
    /// ```
    /// value = sim.peek(100)
    /// ```
    fn peek(&mut self, aigpin: usize) -> bool {
        self.sim.peek(aigpin)
    }

    /// Read an output signal value by signal name
    ///
    /// ## Arguments
    ///
    /// * `signal_path` - Signal name (e.g., "output" or "cpu/alu/result[0]")
    ///
    /// ## Returns
    ///
    /// Signal value as a 64-bit unsigned integer
    ///
    /// ## Raises
    ///
    /// * `RuntimeError` - If signal not found
    ///
    /// ## Example
    ///
    /// ```
    /// value = sim.peek_signal("cpu/output[7:0]")
    /// print(f"Output: 0x{value:02x}")
    /// ```
    fn peek_signal(&mut self, signal_path: &str) -> PyResult<u64> {
        let signal_path: SignalPath = signal_path.parse()?;
        Ok(self.sim.peek_signal(&signal_path)?.0)
    }

    /// Read multiple output signal values at once
    ///
    /// This is more efficient than calling peek() multiple times.
    /// Returns a 64-bit value where bit i corresponds to aigpins[i].
    ///
    /// ## Arguments
    ///
    /// * `aigpins` - List of AIG pin numbers (with invert bit in LSB)
    ///
    /// ## Returns
    ///
    /// 64-bit value where bit i corresponds to aigpins[i]
    ///
    /// ## Raises
    ///
    /// * `RuntimeError` - If pins not found
    /// * `ValueError` - If more than 64 pins are provided
    ///
    /// ## Example
    ///
    /// ```
    /// # Read pins 20,21,22,23
    /// value = sim.peek_bits([20, 21, 22, 23])
    /// print(f"Output: {value:04b}")
    /// ```
    fn peek_bits(&mut self, aigpins: Vec<usize>) -> PyResult<u64> {
        if aigpins.len() > 64 {
            return Err(PyValueError::new_err(
                "Cannot peek more than 64 bits at once",
            ));
        }
        Ok(self.sim.peek_bits(&aigpins)?)
    }

    /// Synchronize GPU operations (wait for completion)
    ///
    /// This ensures all GPU operations have completed before proceeding.
    /// Useful for timing measurements or before accessing results.
    ///
    /// ## Example
    ///
    /// ```
    /// sim.step()
    /// sim.sync()
    /// ```
    fn sync(&self) {
        self.sim.sync();
    }

    /// Get the number of input pins in the circuit
    ///
    /// ## Returns
    ///
    /// Number of input pins
    ///
    /// ## Example
    ///
    /// ```
    /// num_inputs = sim.num_inputs()
    /// print(f"Circuit has {num_inputs} inputs")
    /// ```
    fn num_inputs(&self) -> usize {
        self.sim.num_inputs()
    }

    /// Get the number of output pins in the circuit
    ///
    /// ## Returns
    ///
    /// Number of output pins
    ///
    /// ## Example
    ///
    /// ```
    /// num_outputs = sim.num_outputs()
    /// print(f"Circuit has {num_outputs} outputs")
    /// ```
    fn num_outputs(&self) -> usize {
        self.sim.num_outputs()
    }

    /// Context manager entry
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Context manager exit - ensures GPU sync on exit
    fn __exit__(
        &mut self,
        _exc_type: Option<&PyAny>,
        _exc_value: Option<&PyAny>,
        _traceback: Option<&PyAny>,
    ) -> PyResult<bool> {
        self.sync();
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!(
            "Simulator(inputs={}, outputs={}, initialized={})",
            self.sim.num_inputs(),
            self.sim.num_outputs(),
            self.initialized
        )
    }
}

/// gem-harness - GPU-accelerated hardware simulator
///
/// This module provides Python bindings for the hardware simulator,
/// which simulates digital circuits represented as And-Inverter Graphs (AIGs)
/// on CUDA GPUs.
#[pymodule]
fn harness(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySimulator>()?;
    Ok(())
}
