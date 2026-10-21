// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! FFI bindings for InteractiveSimulator
//!
//! This module provides C-compatible FFI bindings with compile-time safety in
//! Rust and runtime state checking at the FFI boundary.
//!
//! ## Architecture Overview
//!
//! This implementation implements a safe FFI with typestate:
//!
//! ### 1. Internal Rust API
//!
//! The [`internal::SimHandle<State>`] type uses `#[repr(transparent)]` with
//! `PhantomData<State>` to provide **zero-cost compile-time** state tracking.
//!
//! **Key properties:**
//! - `SimHandle<Ready>` and `SimHandle<Poked>` have **identical memory layout**
//! - The `State` type parameter exists **only at compile-time** (zero runtime
//!   cost)
//! - Invalid transitions are caught by the type checker, not at runtime
//!
//! ### 2. C FFI (Runtime Safety)
//!
//! The C interface uses:
//! - [`OpaqueSimHandle`] - opaque pointer for C code (no state information
//!   visible)
//! - `RuntimeHandle` - internal struct with runtime `state: SimState` tracking
//! - All FFI functions validate state at runtime (C can't enforce typestate)
//!
//! ### 3. FFI Boundary Pattern
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │ Rust Side (Compile-Time)                                 │
//! │                                                           │
//! │  SimHandle<Ready>  ──as_raw()──►  *mut Simulator        │
//! │  SimHandle<Poked>  ──as_raw()──►  *mut Simulator        │
//! │       ▲                                │                 │
//! │       │                                │                 │
//! │       └────────────from_raw()──────────┘                 │
//! │                                                           │
//! │  Note: All SimHandle<State> have same memory layout!     │
//! └──────────────────────────────────────────────────────────┘
//!                            │
//!                            ▼
//! ┌──────────────────────────────────────────────────────────┐
//! │ C Side (Runtime)                                         │
//! │                                                           │
//! │  OpaqueSimHandle* ──────► struct (no state info)        │
//! │                                                           │
//! │  C sees normal pointers, no Rust type information        │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## State Transitions
//!
//! ```text
//!                     ┌─────────────┐
//!                     │Uninitialized│
//!                     └──────┬──────┘
//!                            │ create
//!                            ▼
//!                     ┌─────────────┐
//!                     │ Initialized │
//!                     └──────┬──────┘
//!                            │ reset
//!                            ▼
//!                      ┌──────────┐
//!                      │  Ready   │
//!                      └────┬─────┘
//!                           │ poke
//!                           ▼
//!                      ┌──────────┐
//!                      │  Poked   │◄─────┐
//!                      └────┬─────┘      │
//!                           │            │
//!                           ├─ poke ─────┤
//!                      step │            │
//!                           ▼            │ next_cycle
//!                      ┌──────────┐      │
//!                      │ Stepped  │◄──┐  │
//!                      └────┬─────┘   │  │
//!                           │         │  │
//!                           ├─ peek ──┘  │
//!                           │            │
//!                           └────────────┘
//! ```
//!
//! ## C FFI Functions
//!
//! The C interface uses opaque handles and runtime state checks. All functions
//! return a [`SimResult`] struct that includes the current state and any error
//! message.
#![allow(dead_code)]

use std::{
    ffi::{CString, c_char},
    ptr::NonNull,
};

// Expose the typestate API for Rust users
pub use internal::{
    Initialized, Poked, Ready, SimHandle, Stepped, Uninitialized,
};

use crate::{
    signals::SignalError,
    sim::{InteractiveSimulator, SimulatorError},
};

impl From<SimulatorError> for FfiError {
    fn from(error: SimulatorError) -> FfiError {
        FfiError::SimError(error.to_string())
    }
}

impl From<SignalError> for FfiError {
    fn from(error: SignalError) -> FfiError {
        FfiError::SimError(error.to_string())
    }
}

mod internal;

/// Leak a C string for error messages (caller must not free)
fn leak_c_string(s: &str) -> *const c_char {
    CString::new(s)
        .unwrap_or_else(|_| {
            CString::new("error message contains null byte").unwrap()
        })
        .into_raw() as *const c_char
}

/// Internal error type for FFI operations
enum FfiError {
    NullHandle,
    NullPointer(&'static str),
    InvalidUtf8(&'static str),
    InvalidState {
        func_name: &'static str,
        current: SimState,
        expected: String,
    },
    SimError(String),
}

impl FfiError {
    /// Convert error to SimResult
    fn into_sim_result(self, handle: *mut OpaqueSimHandle) -> SimResult {
        match self {
            FfiError::NullHandle => SimResult {
                handle: std::ptr::null_mut(),
                state: SimState::Error,
                error_msg: leak_c_string("null handle provided"),
            },
            FfiError::NullPointer(name) => SimResult {
                handle,
                state: SimState::Error,
                error_msg: leak_c_string(&format!(
                    "null {} pointer provided",
                    name
                )),
            },
            FfiError::InvalidUtf8(name) => SimResult {
                handle,
                state: SimState::Error,
                error_msg: leak_c_string(&format!("invalid UTF-8 in {}", name)),
            },
            FfiError::InvalidState {
                func_name,
                current,
                expected,
            } => {
                let msg = format!(
                    "{} must be called in {} state, currently in {:?}",
                    func_name, expected, current
                );
                SimResult {
                    handle,
                    state: SimState::Error,
                    error_msg: leak_c_string(&msg),
                }
            }
            FfiError::SimError(msg) => SimResult {
                handle,
                state: SimState::Error,
                error_msg: leak_c_string(&msg),
            },
        }
    }
}

/// Opaque handle for C code (type-erases the state)
#[repr(C)]
pub struct OpaqueSimHandle {
    _private: [u8; 0],
}

/// Runtime state tracking for FFI boundary
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SimState {
    Uninitialized = 0,
    Initialized = 1,
    Ready = 2,
    Poked = 3,
    Stepped = 4,
    Error = 255,
}

/// Result type returned by all FFI functions
#[repr(C)]
pub struct SimResult {
    /// Opaque handle to the simulator (null on fatal error)
    pub handle: *mut OpaqueSimHandle,
    /// Current state of the simulator
    pub state: SimState,
    /// Error message (null if no error, caller must NOT free this)
    pub error_msg: *const c_char,
}

/// Internal runtime handle that tracks state dynamically
struct RuntimeHandle {
    sim: NonNull<InteractiveSimulator>,
    state: SimState,
}

/// Create a new simulator instance
///
/// # Parameters
/// - `netlist_path`: Path to Verilog netlist file
/// - `gemparts_path`: Path to GEM partitions file
/// - `num_blocks`: Number of CUDA blocks to use
/// - `device_id`: CUDA device ID (0 for first GPU)
///
/// # Returns
/// [`SimResult`] with [`SimState::Initialized`] on success, or
/// [`SimState::Error`] on failure
///
/// # Safety
/// - `netlist_path` and `gemparts_path` must be valid null-terminated C strings
/// - Returned handle must be freed with [`sim_destroy`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_create(
    netlist_path: *const c_char,
    gemparts_path: *const c_char,
    num_blocks: usize,
    device_id: u8,
) -> SimResult {
    // SAFETY: outer unsafe fn contract guarantees pointer arguments are valid
    // null-terminated C strings.
    unsafe {
        internal::sim_create(netlist_path, gemparts_path, num_blocks, device_id)
    }
    .unwrap_or_else(|e| e.into_sim_result(std::ptr::null_mut()))
}

/// Reset the simulator to initial state
///
/// # Parameters
/// - `handle`: Simulator handle from [`sim_create`]
///
/// # Returns
/// [`SimResult`] with [`SimState::Ready`] on success
///
/// # Valid States
/// Can only be called in [`SimState::Initialized`] state
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_reset(handle: *mut OpaqueSimHandle) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle
    // from sim_create.
    unsafe { internal::sim_reset(handle) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Set an input signal value
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `aigpin`: AIG pin number (without invert bit)
/// - `value`: Signal value (true=high, false=low)
///
/// # Returns
/// [`SimResult`] with [`SimState::Poked`] on success
///
/// # Valid States
/// Can be called in [`SimState::Ready`] or [`SimState::Poked`] states (can poke
/// multiple inputs)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_poke(
    handle: *mut OpaqueSimHandle,
    aigpin: usize,
    value: bool,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle
    // from sim_create.
    unsafe { internal::sim_poke(handle, aigpin, value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Set an input signal value by name
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `signal_path`: Signal name like "input_a" or "cpu/alu/in\[0\]"
/// - `value`: Signal value (true=high, false=low)
///
/// # Returns
/// [`SimResult`] with [`SimState::Poked`] on success
///
/// # Valid States
/// Can be called in [`SimState::Ready`] or [`SimState::Poked`] states
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `signal_path` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_poke_signal(
    handle: *mut OpaqueSimHandle,
    signal_path: *const c_char,
    value: bool,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle and
    // signal_path a valid C string.
    unsafe { internal::sim_poke_signal(handle, signal_path, value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Execute one simulation cycle
///
/// # Parameters
/// - `handle`: Simulator handle
///
/// # Returns
/// [`SimResult`] with [`SimState::Stepped`] on success
///
/// # Valid States
/// Can only be called in [`SimState::Poked`] state (after setting inputs)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_step(handle: *mut OpaqueSimHandle) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle
    // from sim_create.
    unsafe { internal::sim_step(handle) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Read an output signal value
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `aigpin`: AIG pin number (with invert bit in LSB)
/// - `out_value`: Pointer to write output value
///
/// # Returns
/// [`SimResult`] with [`SimState::Stepped`] (can peek multiple outputs)
///
/// # Valid States
/// Can only be called in [`SimState::Stepped`] state (after step)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `out_value` must be a valid pointer to a bool
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_peek(
    handle: *mut OpaqueSimHandle,
    aigpin: usize,
    out_value: *mut bool,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle and
    // out_value a valid writable pointer.
    unsafe { internal::sim_peek(handle, aigpin, out_value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Read an output signal value by name
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `signal_path`: Signal name like "output_z" or "cpu/alu/result\[0\]"
/// - `out_value`: Pointer to write output value
///
/// # Returns
/// [`SimResult`] with [`SimState::Stepped`]
///
/// # Valid States
/// Can only be called in [`SimState::Stepped`] state (after step)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `signal_path` must be a valid null-terminated C string
/// - `out_value` must be a valid pointer to a bool
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_peek_signal(
    handle: *mut OpaqueSimHandle,
    signal_path: *const c_char,
    out_value: *mut u64,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle,
    // signal_path a valid C string, and out_value a valid writable pointer.
    unsafe { internal::sim_peek_signal(handle, signal_path, out_value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Synchronize GPU operations (wait for completion)
///
/// # Parameters
/// - `handle`: Simulator handle
///
/// # Returns
/// [`SimResult`] with unchanged state
///
/// # Valid States
/// Can be called in [`SimState::Stepped`] state
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_sync(handle: *mut OpaqueSimHandle) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle
    // from sim_create.
    unsafe { internal::sim_sync(handle) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Prepare for next simulation cycle
///
/// # Parameters
/// - `handle`: Simulator handle
///
/// # Returns
/// [`SimResult`] with [`SimState::Poked`]
///
/// # Valid States
/// Can only be called in [`SimState::Stepped`] state (after peeking outputs)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_next_cycle(
    handle: *mut OpaqueSimHandle,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle
    // from sim_create.
    unsafe { internal::sim_next_cycle(handle) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Get number of input pins
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_num_inputs(handle: *mut OpaqueSimHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: handle is non-null (checked above); it was created via
    // Box::into_raw::<RuntimeHandle> in sim_create.
    unsafe {
        let h = &*(handle as *mut RuntimeHandle);
        h.sim.as_ref().num_inputs()
    }
}

/// Get number of output pins
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_num_outputs(
    handle: *mut OpaqueSimHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: handle is non-null (checked above); it was created via
    // Box::into_raw::<RuntimeHandle> in sim_create.
    unsafe {
        let h = &*(handle as *mut RuntimeHandle);
        h.sim.as_ref().num_outputs()
    }
}

/// Set multiple input signal values at once
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `aigpins`: Array of AIG pin numbers (without invert bit)
/// - `num_pins`: Number of pins in the array (max 64)
/// - `value`: 64-bit value where bit i corresponds to aigpins\[i\]
///
/// # Returns
/// [`SimResult`] with [`SimState::Poked`] on success
///
/// # Valid States
/// Can be called in [`SimState::Ready`] or [`SimState::Poked`] states
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `aigpins` must point to a valid array of at least `num_pins` elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_poke_bits(
    handle: *mut OpaqueSimHandle,
    aigpins: *const usize,
    num_pins: usize,
    value: u64,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle and
    // aigpins to point to at least num_pins elements.
    unsafe { internal::sim_poke_bits(handle, aigpins, num_pins, value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Read multiple output signal values at once
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `aigpins`: Array of AIG pin numbers (with invert bit in LSB)
/// - `num_pins`: Number of pins in the array (max 64)
/// - `out_value`: Pointer to write 64-bit output where bit i corresponds to
///   aigpins\[i\]
///
/// # Returns
/// [`SimResult`] with [`SimState::Stepped`] on success
///
/// # Valid States
/// Can only be called in [`SimState::Stepped`] state (after step)
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `aigpins` must point to a valid array of at least `num_pins` elements
/// - `out_value` must be a valid pointer to a u64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_peek_bits(
    handle: *mut OpaqueSimHandle,
    aigpins: *const usize,
    num_pins: usize,
    out_value: *mut u64,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle, aigpins (num_pins
    // elements), and out_value to all be valid.
    unsafe { internal::sim_peek_bits(handle, aigpins, num_pins, out_value) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Add signals to trace during simulation
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `signal_paths`: Array of null-terminated C strings with signal paths
/// - `num_signals`: Number of signals in the array
///
/// # Returns
/// [`SimResult`] with unchanged state on success
///
/// # Valid States
/// Can be called in any state
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `signal_paths` must point to a valid array of at least `num_signals`
///   pointers
/// - Each pointer in `signal_paths` must be a valid null-terminated C string
///
/// # Example
/// ```c
/// const char *signals[] = {"sum", "carry"};
/// sim_trace_signals(handle, signals, 2);
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_trace_signals(
    handle: *mut OpaqueSimHandle,
    signal_paths: *const *const c_char,
    num_signals: usize,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be valid,
    // signal_paths to point to num_signals valid C strings.
    unsafe { internal::sim_trace_signals(handle, signal_paths, num_signals) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Dump all traced signals to a VCD file
///
/// # Parameters
/// - `handle`: Simulator handle
/// - `filename`: Path to output VCD file (null-terminated C string)
///
/// # Returns
/// [`SimResult`] with unchanged state on success
///
/// # Valid States
/// Can be called in any state
///
/// # Limitations
/// - This function should only be called ONCE at the end of simulation
/// - Calling it multiple times will overwrite the previous VCD file
/// - All traced signal changes are accumulated in memory until this is called
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - `filename` must be a valid null-terminated C string
///
/// # Example
/// ```c
/// // Run simulation...
/// // Call sim_dump once at the end
/// sim_dump(handle, "output.vcd");
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_dump(
    handle: *mut OpaqueSimHandle,
    filename: *const c_char,
) -> SimResult {
    // SAFETY: outer unsafe fn contract requires handle to be a valid handle and
    // filename a valid null-terminated C string.
    unsafe { internal::sim_dump(handle, filename) }
        .unwrap_or_else(|e| e.into_sim_result(handle))
}

/// Destroy the simulator and free all resources
///
/// # Parameters
/// - `handle`: Simulator handle from [`sim_create`]
///
/// # Safety
/// - `handle` must be a valid handle from [`sim_create`]
/// - After calling this, `handle` must not be used again
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sim_destroy(handle: *mut OpaqueSimHandle) {
    if !handle.is_null() {
        // SAFETY: handle is non-null (checked above); RuntimeHandle and its sim
        // field were both created via Box::into_raw in sim_create.
        unsafe {
            let h = Box::from_raw(handle as *mut RuntimeHandle);
            let _ = Box::from_raw(h.sim.as_ptr());
        }
    }
}
