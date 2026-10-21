// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Internal implementation with typestate pattern
use std::{
    ffi::{CStr, c_char},
    marker::PhantomData,
    path::PathBuf,
    ptr::NonNull,
};

use ulib::Device;

use crate::{
    ffi::{FfiError, OpaqueSimHandle, RuntimeHandle, SimResult, SimState},
    signals::SignalPath,
    sim::{InteractiveSimulator, PeekResult, SimulatorBuilder, SimulatorError},
};

// State Markers

/// Initial state after creation
pub struct Uninitialized;

/// Simulator created but not yet reset
pub struct Initialized;

/// After reset, ready to accept input pokes
pub struct Ready;

/// After poking inputs, can step or poke more
pub struct Poked;

/// After stepping, can peek outputs
pub struct Stepped;

// Typestate Handle

/// Type-safe handle with compile-time state checking.
///
/// Uses `#[repr(transparent)]` to ensure `SimHandle<State>` has identical
/// memory layout to `NonNull<InteractiveSimulator>` regardless of `State`.
/// The `State` parameter exists ONLY at compile-time via `PhantomData`.
#[repr(transparent)]
pub struct SimHandle<State> {
    sim: NonNull<InteractiveSimulator>,
    _state: PhantomData<State>,
}

impl<State> SimHandle<State> {
    /// Extract raw pointer for FFI (kernel-style `as_raw()`)
    #[inline]
    pub(crate) fn as_raw(&self) -> *mut InteractiveSimulator {
        self.sim.as_ptr()
    }

    /// Reconstruct handle from raw pointer (kernel-style `from_raw()`)
    ///
    /// # Safety
    /// - `ptr` must be a valid pointer from `as_raw()`
    /// - Caller must ensure the state parameter matches reality
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *mut InteractiveSimulator) -> Self {
        Self {
            // SAFETY: caller's safety contract requires ptr to be non-null
            // (from as_raw()).
            sim: unsafe { NonNull::new_unchecked(ptr) },
            _state: PhantomData,
        }
    }

    /// Get immutable reference to simulator
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid
    #[inline]
    unsafe fn sim(&self) -> &InteractiveSimulator {
        // SAFETY: self.sim is a valid, aligned pointer created via
        // Box::into_raw in SimHandle::new.
        unsafe { self.sim.as_ref() }
    }

    /// Get mutable reference to simulator
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid
    #[inline]
    unsafe fn sim_mut(&mut self) -> &mut InteractiveSimulator {
        // SAFETY: self.sim is a valid, aligned pointer created via
        // Box::into_raw in SimHandle::new.
        unsafe { self.sim.as_mut() }
    }
}

// State Transitions

impl SimHandle<Uninitialized> {
    /// Create a new simulator handle (internal, not exposed to FFI directly)
    pub fn new(sim: InteractiveSimulator) -> Self {
        Self {
            sim: NonNull::new(Box::into_raw(Box::new(sim))).unwrap(),
            _state: PhantomData,
        }
    }

    /// Transition to Initialized state
    pub fn initialize(self) -> SimHandle<Initialized> {
        SimHandle {
            sim: self.sim,
            _state: PhantomData,
        }
    }
}

impl SimHandle<Initialized> {
    /// Reset the simulator, transitioning to Ready state
    pub fn reset(mut self) -> SimHandle<Ready> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.reset();
        SimHandle {
            sim: self.sim,
            _state: PhantomData,
        }
    }
}

impl SimHandle<Ready> {
    /// Poke an input signal, transitioning to Poked state
    pub fn poke(mut self, aigpin: usize, value: bool) -> SimHandle<Poked> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke(aigpin, value);
        SimHandle {
            sim: self.sim,
            _state: PhantomData,
        }
    }

    /// Poke input signal by name
    pub fn poke_signal(
        mut self,
        signal_path: &SignalPath,
        value: bool,
    ) -> Result<SimHandle<Poked>, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke_signal(signal_path, value)?;
        Ok(SimHandle {
            sim: self.sim,
            _state: PhantomData,
        })
    }

    /// Poke multiple bits at once
    pub fn poke_bits(
        mut self,
        aigpins: &[usize],
        value: u64,
    ) -> Result<SimHandle<Poked>, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke_bits(aigpins, value)?;
        Ok(SimHandle {
            sim: self.sim,
            _state: PhantomData,
        })
    }
}

impl SimHandle<Poked> {
    /// Poke another input signal (can be called multiple times)
    pub fn poke(mut self, aigpin: usize, value: bool) -> SimHandle<Poked> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke(aigpin, value);
        self
    }

    /// Poke input signal by name
    pub fn poke_signal(
        mut self,
        signal_path: &SignalPath,
        value: bool,
    ) -> Result<SimHandle<Poked>, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke_signal(signal_path, value)?;
        Ok(self)
    }

    /// Poke multiple bits at once
    pub fn poke_bits(
        mut self,
        aigpins: &[usize],
        value: u64,
    ) -> Result<SimHandle<Poked>, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.poke_bits(aigpins, value)?;
        Ok(self)
    }

    /// Execute one simulation cycle, transitioning to Stepped state
    pub fn step(mut self) -> SimHandle<Stepped> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.step();
        SimHandle {
            sim: self.sim,
            _state: PhantomData,
        }
    }
}

impl SimHandle<Stepped> {
    /// Read an output signal value (can peek multiple times)
    pub fn peek(&mut self, aigpin: usize) -> bool {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.peek(aigpin)
    }

    /// Read output signal by name
    pub fn peek_signal(
        &mut self,
        signal_path: &SignalPath,
    ) -> Result<PeekResult, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.peek_signal(signal_path)
    }

    /// Read multiple bits at once
    pub fn peek_bits(
        &mut self,
        aigpins: &[usize],
    ) -> Result<u64, SimulatorError> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim_mut() }.peek_bits(aigpins)
    }

    /// Synchronize GPU operations
    pub fn sync(&self) {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim() }.sync();
    }

    /// Transition back to Poked state to continue simulation
    pub fn next_cycle(self) -> SimHandle<Poked> {
        SimHandle {
            sim: self.sim,
            _state: PhantomData,
        }
    }
}

// Common operations available in any state
impl<State> SimHandle<State> {
    /// Get number of input pins
    pub fn num_inputs(&self) -> usize {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim() }.num_inputs()
    }

    /// Get number of output pins
    pub fn num_outputs(&self) -> usize {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim() }.num_outputs()
    }

    /// Add signals to trace by name. Unrecognized names are logged and skipped.
    pub fn trace_by_name(&mut self, signals: &[String]) {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        let sim = unsafe { self.sim_mut() };
        let handles: Vec<_> = signals
            .iter()
            .filter_map(|name| {
                let handle = sim.wire(name);
                if handle.is_none() {
                    tracing::warn!(
                        "Signal {name} not found in traceable wires"
                    );
                }
                handle
            })
            .collect();
        sim.trace(&handles);
    }

    /// Write traced signals as a VCD file.
    pub fn write_vcd<W: std::io::Write>(
        &self,
        writer: W,
        module_name: &str,
    ) -> Result<(), String> {
        // SAFETY: self.sim is a valid pointer created via Box::into_raw in
        // SimHandle::new.
        unsafe { self.sim() }
            .write_vcd(writer, module_name)
            .map_err(|e| format!("{e:?}"))
    }
}

// All states can be dropped safely
impl<State> Drop for SimHandle<State> {
    fn drop(&mut self) {
        // SAFETY: self.sim was created via Box::into_raw in SimHandle::new and
        // is dropped here exactly once.
        unsafe {
            let _ = Box::from_raw(self.sim.as_ptr());
        }
    }
}

// FFI Validation Helpers (Runtime Checks at FFI Boundary)

/// Validate that a handle is non-null and return a mutable reference
unsafe fn validate_handle(
    handle: *mut OpaqueSimHandle,
) -> Result<&'static mut RuntimeHandle, FfiError> {
    if handle.is_null() {
        return Err(FfiError::NullHandle);
    }
    // SAFETY: handle is non-null (checked above); it was created via
    // Box::into_raw::<RuntimeHandle> in sim_create.
    Ok(unsafe { &mut *(handle as *mut RuntimeHandle) })
}

/// Validate that a pointer is non-null
unsafe fn validate_ptr<T>(
    ptr: *const T,
    name: &'static str,
) -> Result<(), FfiError> {
    if ptr.is_null() {
        return Err(FfiError::NullPointer(name));
    }
    Ok(())
}

/// Convert a C string to a Rust String with validation
unsafe fn c_str_to_string(
    ptr: *const c_char,
    name: &'static str,
) -> Result<String, FfiError> {
    // SAFETY: validate_ptr only checks nullness; caller is an unsafe fn that
    // owns this pointer.
    unsafe { validate_ptr(ptr, name) }?;
    // SAFETY: ptr is non-null (validate_ptr succeeded); outer unsafe fn
    // contract requires a valid null-terminated C string.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(String::from)
        .map_err(|_| FfiError::InvalidUtf8(name))
}

/// Check that the handle is in one of the expected states
fn check_state_any(
    h: &RuntimeHandle,
    expected: &[SimState],
    func_name: &'static str,
) -> Result<(), FfiError> {
    if !expected.contains(&h.state) {
        let expected_str = expected
            .iter()
            .map(|s| format!("{:?}", s))
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(FfiError::InvalidState {
            func_name,
            current: h.state,
            expected: expected_str,
        });
    }
    Ok(())
}

// FFI Implementation (Runtime State Tracking)

/// Internal implementation of sim_create
pub(crate) unsafe fn sim_create(
    netlist_path: *const c_char,
    gemparts_path: *const c_char,
    num_blocks: usize,
    device_id: u8,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires netlist_path to be a valid
    // null-terminated C string.
    let netlist_path = PathBuf::from(unsafe {
        c_str_to_string(netlist_path, "netlist_path")
    }?);
    // SAFETY: outer unsafe fn contract requires gemparts_path to be a valid
    // null-terminated C string.
    let gemparts_path = PathBuf::from(unsafe {
        c_str_to_string(gemparts_path, "gemparts_path")
    }?);

    // Build simulator
    let sim = SimulatorBuilder::new(netlist_path, gemparts_path, num_blocks)
        .device(Device::CUDA(device_id))
        .build()
        .map_err(|e| {
            FfiError::SimError(format!("failed to build simulator: {}", e))
        })?;

    // Create runtime handle
    let handle = Box::into_raw(Box::new(RuntimeHandle {
        sim: NonNull::new(Box::into_raw(Box::new(sim))).expect("ptr is null!"),
        state: SimState::Initialized,
    }));

    Ok(SimResult {
        handle: handle as *mut OpaqueSimHandle,
        state: SimState::Initialized,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_reset
pub(crate) unsafe fn sim_reset(
    handle: *mut OpaqueSimHandle,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Initialized], "reset")?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_mut() }.reset();
    h.state = SimState::Ready;

    Ok(SimResult {
        handle,
        state: SimState::Ready,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_poke
pub(crate) unsafe fn sim_poke(
    handle: *mut OpaqueSimHandle,
    aigpin: usize,
    value: bool,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Ready, SimState::Poked], "poke")?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_mut() }.poke(aigpin, value);
    h.state = SimState::Poked;

    Ok(SimResult {
        handle,
        state: SimState::Poked,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_poke_signal
pub(crate) unsafe fn sim_poke_signal(
    handle: *mut OpaqueSimHandle,
    signal_path: *const c_char,
    value: bool,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Ready, SimState::Poked], "poke_signal")?;

    // SAFETY: outer unsafe fn contract requires signal_path to be a valid
    // null-terminated C string.
    let signal_path: SignalPath =
        unsafe { c_str_to_string(signal_path, "signal_path") }?.parse()?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_mut() }.poke_signal(&signal_path, value)?;

    h.state = SimState::Poked;

    Ok(SimResult {
        handle,
        state: SimState::Poked,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_step
pub(crate) unsafe fn sim_step(
    handle: *mut OpaqueSimHandle,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Poked], "step")?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_mut() }.step();
    h.state = SimState::Stepped;

    Ok(SimResult {
        handle,
        state: SimState::Stepped,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_peek
pub(crate) unsafe fn sim_peek(
    handle: *mut OpaqueSimHandle,
    aigpin: usize,
    out_value: *mut bool,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires out_value to be a valid
    // writable pointer.
    unsafe { validate_ptr(out_value, "out_value") }?;
    check_state_any(h, &[SimState::Stepped], "peek")?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    let value = unsafe { h.sim.as_mut() }.peek(aigpin);
    // SAFETY: out_value is non-null (validate_ptr succeeded); outer fn contract
    // requires a valid writable pointer.
    unsafe { *out_value = value };

    Ok(SimResult {
        handle,
        state: SimState::Stepped,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_peek_signal
pub(crate) unsafe fn sim_peek_signal(
    handle: *mut OpaqueSimHandle,
    signal_path: *const c_char,
    out_value: *mut u64,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires out_value to be a valid
    // writable pointer.
    unsafe { validate_ptr(out_value, "out_value") }?;
    check_state_any(h, &[SimState::Stepped], "peek_signal")?;

    // SAFETY: outer unsafe fn contract requires signal_path to be a valid
    // null-terminated C string.
    let signal_path: SignalPath =
        unsafe { c_str_to_string(signal_path, "signal_path") }?.parse()?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    let value = unsafe { h.sim.as_mut() }.peek_signal(&signal_path)?;

    // SAFETY: out_value is non-null (validate_ptr succeeded); outer fn contract
    // requires a valid writable pointer.
    unsafe { *out_value = value.0 };

    Ok(SimResult {
        handle,
        state: SimState::Stepped,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_sync
pub(crate) unsafe fn sim_sync(
    handle: *mut OpaqueSimHandle,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Stepped], "sync")?;

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_ref() }.sync();

    Ok(SimResult {
        handle,
        state: h.state,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_next_cycle that can use ?
pub(crate) unsafe fn sim_next_cycle(
    handle: *mut OpaqueSimHandle,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    check_state_any(h, &[SimState::Stepped], "next_cycle")?;

    h.state = SimState::Poked;

    Ok(SimResult {
        handle,
        state: SimState::Poked,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_poke_bits
pub(crate) unsafe fn sim_poke_bits(
    handle: *mut OpaqueSimHandle,
    aigpins: *const usize,
    num_pins: usize,
    value: u64,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires aigpins to be a valid pointer
    // to at least num_pins elements.
    unsafe { validate_ptr(aigpins, "aigpins") }?;
    check_state_any(h, &[SimState::Ready, SimState::Poked], "poke_bits")?;

    // SAFETY: aigpins is non-null (validate_ptr succeeded); outer fn contract
    // requires it to point to at least num_pins valid elements.
    let aigpins_slice =
        unsafe { std::slice::from_raw_parts(aigpins, num_pins) };
    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_mut() }.poke_bits(aigpins_slice, value)?;

    h.state = SimState::Poked;

    Ok(SimResult {
        handle,
        state: SimState::Poked,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_peek_bits
pub(crate) unsafe fn sim_peek_bits(
    handle: *mut OpaqueSimHandle,
    aigpins: *const usize,
    num_pins: usize,
    out_value: *mut u64,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires aigpins to be a valid pointer
    // to at least num_pins elements.
    unsafe { validate_ptr(aigpins, "aigpins") }?;
    // SAFETY: outer unsafe fn contract requires out_value to be a valid
    // writable pointer.
    unsafe { validate_ptr(out_value, "out_value") }?;
    check_state_any(h, &[SimState::Stepped], "peek_bits")?;

    // SAFETY: aigpins is non-null (validate_ptr succeeded); outer fn contract
    // requires it to point to at least num_pins valid elements.
    let aigpins_slice =
        unsafe { std::slice::from_raw_parts(aigpins, num_pins) };
    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    let value = unsafe { h.sim.as_mut() }.peek_bits(aigpins_slice)?;

    // SAFETY: out_value is non-null (validate_ptr succeeded); outer fn contract
    // requires a valid writable pointer.
    unsafe { *out_value = value };

    Ok(SimResult {
        handle,
        state: SimState::Stepped,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_trace_signals
pub(crate) unsafe fn sim_trace_signals(
    handle: *mut OpaqueSimHandle,
    signal_paths: *const *const c_char,
    num_signals: usize,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires signal_paths to be a valid
    // pointer to at least num_signals elements.
    unsafe { validate_ptr(signal_paths, "signal_paths") }?;

    // SAFETY: signal_paths is non-null (validate_ptr succeeded); outer fn
    // contract requires it to point to at least num_signals valid pointers.
    let signal_paths_slice =
        unsafe { std::slice::from_raw_parts(signal_paths, num_signals) };
    let mut signals = Vec::with_capacity(num_signals);

    for &signal_path_ptr in signal_paths_slice {
        // SAFETY: outer fn contract requires each pointer in signal_paths to be
        // a valid null-terminated C string.
        let signal =
            unsafe { c_str_to_string(signal_path_ptr, "signal_path") }?;
        signals.push(signal);
    }

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    let sim = unsafe { h.sim.as_mut() };
    let handles: Vec<_> = signals
        .iter()
        .filter_map(|name| {
            let handle = sim.wire(name);
            if handle.is_none() {
                tracing::warn!("Signal {name} not found in traceable wires");
            }
            handle
        })
        .collect();
    sim.trace(&handles);

    Ok(SimResult {
        handle,
        state: h.state,
        error_msg: std::ptr::null(),
    })
}

/// Internal implementation of sim_dump
pub(crate) unsafe fn sim_dump(
    handle: *mut OpaqueSimHandle,
    filename: *const c_char,
) -> Result<SimResult, FfiError> {
    // SAFETY: outer unsafe fn contract requires handle to be a valid
    // RuntimeHandle pointer from sim_create.
    let h = unsafe { validate_handle(handle) }?;
    // SAFETY: outer unsafe fn contract requires filename to be a valid
    // null-terminated C string.
    let filename = unsafe { c_str_to_string(filename, "filename") }?;

    use std::{fs::File, io::BufWriter};

    let file = File::create(&filename).map_err(|e| {
        FfiError::SimError(format!("failed to create VCD file: {}", e))
    })?;
    let buf = BufWriter::new(file);

    // SAFETY: validate_handle verified handle is valid; h.sim was initialized
    // in sim_create.
    unsafe { h.sim.as_ref() }
        .write_vcd(buf, "top_module")
        .map_err(|e| {
            FfiError::SimError(format!("failed to write VCD: {}", e))
        })?;

    Ok(SimResult {
        handle,
        state: h.state,
        error_msg: std::ptr::null(),
    })
}
