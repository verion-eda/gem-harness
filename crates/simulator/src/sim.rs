// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Interactive cycle-by-cycle GPU simulator library

use std::{
    collections::HashMap,
    fmt::{self, Debug, Display},
    path::Path,
    sync::Arc,
};

use aig::{AIG, DriverType};
use anyhow::Context;
use flatten::FlattenedScriptV1;
// CUDA bindings from gem-cuda crate
#[cfg(feature = "cuda")]
use gem_cuda::ucci;
use gem_cuda::ucci::{ClockEdgeInfo, WatchCondition};
use indexmap::IndexMap;
use netlistdb::{GeneralPinName, NetlistDB};
use staging::StagedAIG;
#[cfg(feature = "cuda")]
use ulib::AsUPtrMut;
use ulib::{Device, UVec};

#[cfg(feature = "cuda")]
use crate::pipe::GpuDpiPipe;
use crate::{
    cpu_simulator::CpuSimulator,
    dpi::{self, DpiCallInfo, DpiMode, DpiState},
    offset::StateOffset,
    signals::SignalPath,
    vcd::{Tracer, TracerBuilder, WireHandle},
};

#[derive(Debug)]
pub enum SimulatorError {
    InvalidSignal(String),
    PinNotFound(usize),
}

impl Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PinNotFound(pin) => {
                write!(f, "AIG pin {pin} not found in input map")
            }
            Self::InvalidSignal(err) => write!(f, "invalid signal: {err}"),
        }
    }
}

#[cfg(feature = "python")]
impl From<SimulatorError> for pyo3::PyErr {
    fn from(error: SimulatorError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
    }
}

impl From<SimulatorError> for anyhow::Error {
    fn from(value: SimulatorError) -> Self {
        anyhow::anyhow!("{value}")
    }
}

#[cfg(feature = "cuda")]
fn prepare_watches(
    watches: Option<Vec<WatchCondition>>,
    device: Device,
) -> (UVec<u32>, usize, UVec<u32>, usize) {
    // Separate watches into stop and toggle arrays
    let (stop_watches_vec, toggle_watches_vec): (Vec<_>, Vec<_>) = watches
        .unwrap_or_default()
        .into_iter()
        .partition(|w| matches!(w, WatchCondition::Stop(_)));

    // Serialize stop watches: 3 u32s each [offset, mask, expected]
    let num_stop_watches = stop_watches_vec.len();
    let mut stop_watches_uvec: UVec<u32> = UVec::new_zeroed_with_capacity(
        num_stop_watches * 3,
        num_stop_watches * 3,
        device,
    );
    for (i, w) in stop_watches_vec.iter().enumerate() {
        #[allow(clippy::identity_op)]
        if let WatchCondition::Stop(sw) = w {
            stop_watches_uvec[i * 3 + 0] = sw.state_offset;
            stop_watches_uvec[i * 3 + 1] = sw.mask;
            stop_watches_uvec[i * 3 + 2] = sw.expected;
        }
    }

    // Serialize toggle watches: 8 u32s each
    // [offset, mask, start_offset, freq, pe_word, pe_mask, ne_word, ne_mask]
    let num_toggle_watches = toggle_watches_vec.len();
    let mut toggle_watches_uvec: UVec<u32> = UVec::new_zeroed_with_capacity(
        num_toggle_watches * 8,
        num_toggle_watches * 8,
        device,
    );
    for (i, w) in toggle_watches_vec.iter().enumerate() {
        if let WatchCondition::Toggle(tw) = w {
            let arr = tw.to_u32_array();
            for (j, &val) in arr.iter().enumerate() {
                toggle_watches_uvec[i * 8 + j] = val;
            }
        }
    }

    (
        stop_watches_uvec,
        num_stop_watches,
        toggle_watches_uvec,
        num_toggle_watches,
    )
}

/// Configuration for a clock signal to toggle during `batched_step`.
#[derive(Debug, Clone, Copy)]
pub struct ToggleClock<'a> {
    /// Signal name (e.g., "clk_i")
    pub signal: &'a str,
    /// Toggle frequency: signal toggles every `freq` cycles
    pub freq: u32,
    /// Phase offset: first toggle happens at cycle `phase`
    pub phase: u32,
}

/// Interactive simulator for cycle-by-cycle testbenches.
///
/// Built via [`SimulatorBuilder`]. Drive inputs with `poke*`, advance time
/// with [`step`](Self::step) or [`batched_step`](Self::batched_step), and
/// observe outputs with `peek*`. Waveform capture is opt-in via
/// [`trace`](Self::trace) / [`write_vcd`](Self::write_vcd).
pub struct InteractiveSimulator {
    device: Device,
    num_blocks: usize,
    num_major_stages: usize,
    state_size: usize,

    // GPU state buffers - using simple single buffer approach
    // State contains: [inputs + FF outputs + ...] packed as u32 words
    current_state: UVec<u32>,
    next_state: UVec<u32>,
    sram_storage: UVec<u32>,

    // Compiled circuit (immutable after init)
    blocks_start: UVec<usize>,
    blocks_data: UVec<u32>,

    // Mapping between AIG pins and state offsets
    input_map: indexmap::IndexMap<usize, u32>,
    output_map: indexmap::IndexMap<usize, u32>,

    // References for name lookup
    netlistdb: Arc<NetlistDB>,
    aig: Arc<AIG>,

    step_num: u64,
    tracer: Tracer,

    // Pre-computed clock edge info for simulators
    clock_edges: Vec<ClockEdgeInfo>,

    // DPI support
    dpi: DpiState,
    #[cfg(feature = "cuda")]
    dpi_gpu_pipe: GpuDpiPipe,
}

/// Trait for values that can be poked into the simulator
pub trait PokeValue {
    fn to_u64(self) -> u64;
}

impl PokeValue for bool {
    fn to_u64(self) -> u64 {
        if self { 1 } else { 0 }
    }
}

impl PokeValue for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

impl PokeValue for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl PokeValue for i32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

/// Wrapper for peek results to allow easy comparison with different types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeekResult(pub u64);

impl From<u64> for PeekResult {
    fn from(v: u64) -> Self {
        PeekResult(v)
    }
}

impl std::ops::Deref for PeekResult {
    type Target = u64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Display for PeekResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::LowerHex for PeekResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(&self.0, f)
    }
}

impl std::fmt::UpperHex for PeekResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperHex::fmt(&self.0, f)
    }
}

impl PartialEq<bool> for PeekResult {
    fn eq(&self, other: &bool) -> bool {
        (self.0 != 0) == *other
    }
}

impl PartialEq<u64> for PeekResult {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<i32> for PeekResult {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other as u64
    }
}

impl InteractiveSimulator {
    /// Create a new interactive simulator from a compiled GEM script
    pub(crate) fn new(
        script: FlattenedScriptV1,
        num_blocks: usize,
        device: Device,
        netlistdb: Arc<NetlistDB>,
        aig: Arc<AIG>,
        dpi: DpiState,
        #[cfg(feature = "cuda")] dpi_gpu_pipe: GpuDpiPipe,
    ) -> Self {
        let state_size = script.reg_io_state_size as usize;

        // Initialize state buffers
        #[allow(unused_mut)]
        let mut current_state = UVec::new_zeroed(state_size, device);
        let next_state = UVec::new_zeroed(state_size, device);
        let sram_storage =
            UVec::new_zeroed(script.sram_storage_size as usize, device);

        // Ensure buffers are allocated on GPU (only if using GPU backend)
        #[cfg(feature = "cuda")]
        if matches!(device, Device::CUDA(_)) {
            current_state.as_mut_uptr(device);
        }

        let output_map = script.output_map;

        // Build aigpin_iv → preferred display name with explicit priority:
        //   cell pin names < primary port names < keep-net names.
        // Later insertions override earlier ones, so we insert in priority
        // order.
        let mut aigpin_iv_to_name: HashMap<usize, String> = HashMap::new();

        // 1. Cell pin names (lowest priority — e.g. "_00_:Q")
        for (pin_id, &aigpin_iv) in aig.pin2aigpin_iv.iter().enumerate() {
            if aigpin_iv != usize::MAX {
                aigpin_iv_to_name.entry(aigpin_iv).or_insert_with(|| {
                    netlistdb.pinnames[pin_id].dbg_fmt_pin()
                });
            }
        }

        // 2. Primary port names (e.g. "count[22]", "inc[0]", "clk").
        // Output ports are sinks so pin2aigpin_iv[port_pin] = usize::MAX;
        // resolve the driving aigpin_iv through the net's root pin instead.
        for (&(ref port_name, bus_idx), &pin_id) in &netlistdb.portname2pinid {
            let net_id = netlistdb.pin2net[pin_id];
            let root_pin =
                netlistdb.net2pin.items[netlistdb.net2pin.start[net_id]];
            let aigpin_iv = aig.pin2aigpin_iv[root_pin];
            if aigpin_iv == usize::MAX {
                continue;
            }
            let name = format!(
                "{}{}",
                port_name,
                bus_idx.map_or(String::new(), |i| format!("[{i}]")),
            );
            aigpin_iv_to_name.insert(aigpin_iv, name);
        }

        // 3. Keep-net names (highest priority — user-annotated signal names)
        for &net_id in &netlistdb.keep_nets {
            let root_pin =
                netlistdb.net2pin.items[netlistdb.net2pin.start[net_id]];
            let aigpin_iv = aig.pin2aigpin_iv[root_pin];
            if aigpin_iv == usize::MAX {
                continue;
            }
            aigpin_iv_to_name.insert(
                aigpin_iv,
                netlistdb.netnames[net_id].dbg_fmt_pin().to_string(),
            );
        }

        // Register all observable signals. output_map covers computed state
        // (DFF D-inputs, SRAM read-data); input_map covers stored state
        // (DFF Q-outputs) and primary inputs.
        let mut tracer_builder = TracerBuilder::new();

        for (&aigpin_iv, &output_pos) in output_map.iter() {
            if let Some(name) = aigpin_iv_to_name.get(&aigpin_iv) {
                tracer_builder.add_wire(aigpin_iv, name.clone(), output_pos);
            }
        }

        // input_map keys are aigpin without the invert bit.
        for (&aigpin, &state_offset) in script.input_map.iter() {
            let aigpin_iv = aigpin << 1;
            if let Some(name) = aigpin_iv_to_name.get(&aigpin_iv) {
                tracer_builder.add_wire(aigpin_iv, name.clone(), state_offset);
            }
        }

        let keep_handles: Vec<WireHandle> = netlistdb
            .keep_nets
            .iter()
            .filter_map(|&net_id| {
                let name = netlistdb.netnames[net_id].dbg_fmt_pin();
                tracer_builder.wire(&name)
            })
            .collect();

        // Pre-compute clock edge info
        let clock_edges: Vec<ClockEdgeInfo> = aig
            .clock_pin2aigpins
            .iter()
            .filter_map(|(&pin, &(pe, ne))| {
                let aigpin = aig.pin2aigpin_iv[pin] >> 1;
                let clock_pos = *script.input_map.get(&aigpin)?;
                let clock_word = clock_pos >> 5;
                let clock_mask = 1u32 << (clock_pos & 31);

                let (pe_word, pe_mask) = if pe != usize::MAX {
                    if let Some(&p) = script.input_map.get(&pe) {
                        (p >> 5, 1u32 << (p & 31))
                    } else {
                        (u32::MAX, 0)
                    }
                } else {
                    (u32::MAX, 0)
                };

                let (ne_word, ne_mask) = if ne != usize::MAX {
                    if let Some(&p) = script.input_map.get(&ne) {
                        (p >> 5, 1u32 << (p & 31))
                    } else {
                        (u32::MAX, 0)
                    }
                } else {
                    (u32::MAX, 0)
                };

                Some(ClockEdgeInfo {
                    clock_word,
                    clock_mask,
                    posedge_word: pe_word,
                    posedge_mask: pe_mask,
                    negedge_word: ne_word,
                    negedge_mask: ne_mask,
                })
            })
            .collect();

        // Ensure next_state is also on GPU (edge detection reads old values
        // from it)
        #[cfg(feature = "cuda")]
        let mut next_state = next_state;
        #[cfg(feature = "cuda")]
        if matches!(device, Device::CUDA(_)) {
            next_state.as_mut_uptr(device);
        }

        Self {
            device,
            num_blocks,
            num_major_stages: script.num_major_stages,
            state_size,
            current_state,
            next_state,
            sram_storage,
            blocks_start: script.blocks_start.clone(),
            blocks_data: script.blocks_data.clone(),
            input_map: script.input_map,
            output_map,
            netlistdb,
            aig,
            tracer: tracer_builder.build(keep_handles),
            step_num: 0,
            clock_edges,
            dpi,
            #[cfg(feature = "cuda")]
            dpi_gpu_pipe,
        }
    }

    /// Execute a single simulation cycle.
    ///
    /// This is the primary interface for testbenches: poke inputs, call
    /// `step()`, then peek outputs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use simulator::sim::InteractiveSimulator;
    /// # let mut sim: InteractiveSimulator = todo!();
    /// sim.step();
    /// ```
    pub fn step(&mut self) {
        self.run(
            None,
            #[cfg(feature = "cuda")]
            None,
        );
    }

    /// Internal: execute up to `max_cycles` cycles with optional watches.
    ///
    /// On CPU, `max_cycles` must be `None` (single-cycle only). On GPU,
    /// clocks are toggled by the kernel via the supplied toggle watches
    /// (see [`InteractiveSimulator::batched_step`]).
    ///
    /// # Buffer contract
    ///
    /// `current_state` is the testbench-visible buffer. Poke inputs into it
    /// before calling `step()`, and peek outputs from it afterwards.
    ///
    /// Internally, the kernel writes results into `next_state`, which is
    /// pre-seeded with a copy of `current_state` so that:
    ///   1. Un-poked inputs persist across steps (the kernel only writes
    ///      output/FF positions).
    ///   2. Intra-stage feedback reads valid old-state values until overwritten
    ///      by the current stage.
    ///
    /// At the end of `step()`, the buffers are swapped so `current_state`
    /// always holds the latest computed state.
    ///
    /// # Note on synchronisation (CUDA)
    ///
    /// `current_state` is explicitly uploaded CPU -> GPU before the kernel call
    /// (pokes wrote to the CPU side between steps).
    ///
    /// `next_state` is NOT uploaded: its GPU data from the previous kernel run
    /// is the "old state" the kernel reads for clock edge detection. The
    /// cooperative copy inside the kernel then initialises `state_b` from
    /// `state_a` before simulation begins.
    fn run(
        &mut self,
        max_cycles: Option<usize>,
        #[cfg(feature = "cuda")] watches: Option<Vec<WatchCondition>>,
    ) -> u32 {
        let cycles_simulated: u32 = match self.device {
            Device::CPU => {
                assert_eq!(max_cycles, None);
                // Run DPI every-cycle callbacks before CPU simulation so their
                // outputs are visible to the circuit as primary inputs this
                // cycle.
                self.dpi.execute_every_cycle(self.current_state.as_mut());
                CpuSimulator::simulate_single_cycle(
                    self.num_blocks,
                    self.num_major_stages,
                    self.blocks_start.as_ref(),
                    self.blocks_data.as_ref(),
                    self.sram_storage.as_mut(),
                    self.state_size,
                    self.current_state.as_mut(),
                    self.next_state.as_mut(),
                    &self.clock_edges,
                );
                1
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                let max_cycles = max_cycles.unwrap_or(1);

                let mut cycles_retired: UVec<u32> =
                    UVec::new_zeroed_with_capacity(1, 1, self.device);
                let mut stop_reason: UVec<u32> =
                    UVec::new_zeroed_with_capacity(1, 1, self.device);
                cycles_retired[0] = 0;
                stop_reason[0] = 0;
                cycles_retired.as_mut_uptr(self.device);
                stop_reason.as_mut_uptr(self.device);

                let (
                    mut stop_watches_uvec,
                    num_stop_watches,
                    mut toggle_watches_uvec,
                    num_toggle_watches,
                ) = prepare_watches(watches, self.device);
                stop_watches_uvec.as_mut_uptr(self.device);
                toggle_watches_uvec.as_mut_uptr(self.device);

                // Serialize clock edges for GPU
                let num_clock_edges = self.clock_edges.len();
                let mut clock_edges_uvec: UVec<u32> =
                    UVec::new_zeroed_with_capacity(
                        num_clock_edges * 6,
                        num_clock_edges * 6,
                        self.device,
                    );
                for (i, edge) in self.clock_edges.iter().enumerate() {
                    let arr = edge.to_u32_array();
                    for (j, &val) in arr.iter().enumerate() {
                        clock_edges_uvec[i * 6 + j] = val;
                    }
                }
                clock_edges_uvec.as_mut_uptr(self.device);

                // CPU-only every-cycle DPI calls (And inputs, not pipeable).
                // Must run before state upload so their outputs land in
                // current_state before the kernel reads it.
                if self.dpi_gpu_pipe.has_cpu_only_calls() {
                    assert!(
                        max_cycles == 1,
                        "batched GPU steps with CPU-only DPI callbacks are \
                         not supported; ensure all every-cycle DPI inputs are \
                         StateOffset or Constant",
                    );
                    self.dpi.execute_by_indices(
                        &self.dpi_gpu_pipe.cpu_only_indices.clone(),
                        self.current_state.as_mut(),
                    );
                }

                // Upload current_state to GPU (pokes wrote to CPU side)
                self.current_state.as_mut_uptr(self.device);
                // next_state is NOT uploaded: its GPU data from the previous
                // kernel run is the "old state" the kernel
                // needs for edge detection.

                // setup and start poll thread if necessary
                let mut pipe_channel = self
                    .dpi_gpu_pipe
                    .open_pipe_channel(self.device, self.dpi.callbacks.clone());

                ucci::simulate_v1_batches_with_watches(
                    self.num_blocks,
                    self.num_major_stages,
                    &self.blocks_start,
                    &self.blocks_data,
                    &mut self.sram_storage,
                    self.state_size,
                    &mut self.current_state,
                    &mut self.next_state,
                    max_cycles,
                    &mut stop_watches_uvec,
                    num_stop_watches,
                    &mut toggle_watches_uvec,
                    num_toggle_watches,
                    &mut clock_edges_uvec,
                    num_clock_edges,
                    &mut cycles_retired,
                    &mut stop_reason,
                    &mut pipe_channel.pipe_buf,
                    &self.dpi_gpu_pipe.in_offsets,
                    &self.dpi_gpu_pipe.in_port_widths,
                    &self.dpi_gpu_pipe.out_offsets,
                    &self.dpi_gpu_pipe.out_port_widths,
                    &self.dpi_gpu_pipe.call_descs,
                    self.dpi_gpu_pipe.n_calls(),
                    self.device,
                );

                pipe_channel.shutdown();

                cycles_retired[0]
            }
        };

        self.tracer.record_step(
            self.step_num,
            &self.current_state,
            &self.next_state,
        );

        // Swap buffers so next becomes current.
        match self.device {
            Device::CPU => {
                // result is always in next_state, so always swap.
                std::mem::swap(&mut self.current_state, &mut self.next_state);
            }
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                // TODO here it makes it hard to reason about correctness - need
                // to rethink this part to make it much more evident that this
                // is correct.
                //
                // the kernel does N-1 internal ping-pong swaps (skipping the
                // last cycle), so the result ends up in state_b (next_state)
                // when N is odd, or state_a (current_state) when N is even.
                if cycles_simulated % 2 == 1 {
                    std::mem::swap(
                        &mut self.current_state,
                        &mut self.next_state,
                    );
                }
            }
        }

        self.step_num += cycles_simulated as u64;
        cycles_simulated
    }

    /// Set an input signal value (call before step())
    ///
    /// pin_aigpin: AIG pin number (without invert bit)
    /// value: true for high, false for low
    pub fn poke(&mut self, pin_aigpin: usize, value: bool) {
        let Some(&raw) = self.input_map.get(&pin_aigpin) else {
            eprintln!("Warning: AIG pin {} not found in input map", pin_aigpin);
            return;
        };
        self.poke_offset(StateOffset::from(raw), value);
    }

    /// Set multiple input bits at once
    ///
    /// pin_aigpins: slice of AIG pin numbers (without invert bit)
    /// value: value to set
    pub fn poke_bits(
        &mut self,
        pin_aigpins: &[usize],
        value: u64,
    ) -> Result<(), SimulatorError> {
        let offsets = pin_aigpins
            .iter()
            .take(64)
            .map(|&pin| {
                self.input_map
                    .get(&pin)
                    .map(|&o| StateOffset::from(o))
                    .ok_or(SimulatorError::PinNotFound(pin))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.poke_offsets(&offsets, value);
        Ok(())
    }

    // TODO this should actually do the step? or just batched_step that also
    // toggles some signals
    /// Toggle a single-bit signal (call before step()).
    ///
    /// Clocks start low after reset, so the first toggle is a rising edge.
    /// Accepts a `&str` path or a pre-built [`SignalPath`].
    ///
    /// ## Example
    ///
    /// ```no_run
    /// # use simulator::signals::SignalPath;
    /// # use simulator::sim::{InteractiveSimulator, SimulatorError};
    /// # fn main() -> anyhow::Result<()> {
    /// # let mut sim: InteractiveSimulator = todo!();
    /// let clk_sg: SignalPath = "clk".parse()?;
    /// sim.toggle(&clk_sg)?; // 0 -> 1, posedge
    /// sim.step();
    /// sim.toggle(&clk_sg)?; // 1 -> 0, negedge
    /// sim.step();
    /// # Ok(())
    /// # }
    /// ```
    pub fn toggle(
        &mut self,
        signal_path: &SignalPath,
    ) -> Result<(), SimulatorError> {
        let signal = signal_path;
        let pins = self.signal_path_to_aigpins(signal)?;
        if pins.len() != 1 {
            return Err(SimulatorError::InvalidSignal(format!(
                "Signal '{signal}' must be a single bit, got {} bits",
                pins.len()
            )));
        }

        let aigpin = pins[0].0 >> 1;

        // Find the state offset for this signal
        let state_offset = match self.input_map.get(&aigpin) {
            Some(&offset) => StateOffset::from(offset),
            None => {
                return Err(SimulatorError::InvalidSignal(format!(
                    "Signal '{signal}' not found in input map",
                )));
            }
        };

        // Toggle the value. GPU upload is deferred to the next step()'s copy
        // block.
        self.current_state.as_mut()[state_offset.word_idx()] ^=
            1u32 << state_offset.bit_idx();

        Ok(())
    }

    /// Read a signal's current value from the state vector.
    ///
    /// Looks up the signal in two regions of the state vector:
    ///  1. **input region** (`input_map`, keyed by aigpin without invert):
    ///     state-holding sources — primary inputs, clock flags, DFF Q, and DPI
    ///     outputs. Covers all internal signals.
    ///  2. **output region** (`output_map`, keyed by aigpin_iv with invert):
    ///     combinational endpoints wired to module-level output ports. The
    ///     invert bit is already baked into the stored value.
    pub fn peek(&mut self, pin_aigpin: usize) -> bool {
        let aigpin = pin_aigpin >> 1;
        let invert = (pin_aigpin & 1) != 0;

        // Try input region first — covers DFF Q, DPI, primary inputs.
        if let Some(&raw) = self.input_map.get(&aigpin) {
            let val = Self::read_bit(
                self.current_state.as_ref(),
                StateOffset::from(raw),
            );
            return val ^ invert;
        }

        // Combinational module outputs.
        // Key includes the invert bit; stored value is already inverted.
        let Some(&raw) = self.output_map.get(&pin_aigpin) else {
            eprintln!(
                "Warning: AIG pin {} not found in input_map or output_map",
                pin_aigpin
            );
            return false;
        };
        self.peek_offset(StateOffset::from(raw))
    }

    /// Read multiple output bits at once
    ///
    /// pin_aigpins: slice of AIG pin numbers (with invert bit in LSB)
    /// Returns: combined value
    pub fn peek_bits(
        &mut self,
        pin_aigpins: &[usize],
    ) -> Result<u64, SimulatorError> {
        let offsets = pin_aigpins
            .iter()
            .take(64)
            .map(|&pin| {
                self.output_map
                    .get(&pin)
                    .map(|&o| StateOffset::from(o))
                    .ok_or(SimulatorError::PinNotFound(pin))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.peek_offsets(&offsets))
    }

    /// Synchronize GPU - wait for all operations to complete
    /// Ensure all simulator operations complete before reading results.
    /// # Example
    /// ```no_run
    /// # use simulator::sim::InteractiveSimulator;
    /// # let sim: InteractiveSimulator = todo!();
    /// sim.sync();
    /// ```
    pub fn sync(&self) {
        match self.device {
            Device::CPU => {} // CPU is synchronous, no-op
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => {
                self.device.synchronize();
            }
        }
    }

    /// Reset all state to zero
    /// Similar to Verilator's reset logic or manually setting state
    pub fn reset(&mut self) {
        // Zero out current state
        let state_slice = self.current_state.as_mut();
        state_slice.fill(0);

        // Run DPI Once callbacks on zeroed state (before GPU upload)
        self.dpi.execute_once(self.current_state.as_mut());

        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::CUDA(_)) {
            self.current_state.as_mut_uptr(self.device);
        }

        // Zero out next_state (edge detection reads old values from it)
        let next_slice = self.next_state.as_mut();
        next_slice.fill(0);
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::CUDA(_)) {
            self.next_state.as_mut_uptr(self.device);
        }

        // Zero out SRAM
        let sram_slice = self.sram_storage.as_mut();
        sram_slice.fill(0);
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::CUDA(_)) {
            self.sram_storage.as_mut_uptr(self.device);
        }

        self.sync();
    }

    /// Resolve a [`SignalPath`] to AIG pin indices.
    ///
    /// Returns `Ok(Vec<(aigpin_iv, bit_idx)>)` where:
    /// - `aigpin_iv`: AIG pin index with invert bit in LSB
    /// - `bit_idx`: bit position within the bus (0 for scalars)
    ///
    /// If `signal` specifies a bus bit (e.g. `result[3]`), returns a
    /// single-element vec. If no bit index is given, returns all bits sorted
    /// by bit index.
    fn signal_path_to_aigpins(
        &self,
        signal: &SignalPath,
    ) -> Result<Vec<(usize, isize)>, SimulatorError> {
        let hier_path = signal.hier_name();
        let pin_name = signal.pin_name().clone();

        if let Some(idx) = signal.bus_idx() {
            let key = (hier_path, pin_name, Some(idx));
            let pin_id = self
                .netlistdb
                .pinname2id
                .get(&key as &dyn GeneralPinName)
                .ok_or_else(|| {
                    SimulatorError::InvalidSignal(format!(
                        "Signal '{signal}' not found in netlist",
                    ))
                })?;
            let aigpin_iv = self.aig.pin2aigpin_iv[*pin_id];
            Ok(vec![(aigpin_iv, idx)])
        } else {
            // First try as scalar (bus_idx = None).
            let key_scalar = (hier_path.clone(), pin_name.clone(), None);
            if let Some(pin_id) = self
                .netlistdb
                .pinname2id
                .get(&key_scalar as &dyn GeneralPinName)
            {
                let aigpin_iv = self.aig.pin2aigpin_iv[*pin_id];
                return Ok(vec![(aigpin_iv, 0)]);
            }

            // Scan for bus bits (NetlistDB has no secondary index on name
            // alone, so a linear scan is required when no bit index is given).
            let mut found_pins = Vec::new();
            for (i, (h, n, idx_opt)) in
                self.netlistdb.pinnames.iter().enumerate()
            {
                if h == &hier_path
                    && n == &pin_name
                    && let Some(bit_idx) = idx_opt
                {
                    let aigpin_iv = self.aig.pin2aigpin_iv[i];
                    found_pins.push((aigpin_iv, *bit_idx));
                }
            }

            if found_pins.is_empty() {
                return Err(SimulatorError::InvalidSignal(format!(
                    "Signal '{signal}' not found in netlist"
                )));
            }

            found_pins.sort_by_key(|&(_, idx)| idx);
            Ok(found_pins)
        }
    }

    /// Set an input signal value by name (call before step()).
    ///
    /// Scalar and bus signals are both supported; for a bus, bit 0 of `value`
    /// maps to the lowest-indexed bit. Up to 64 bits are written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use simulator::sim::SimulatorError;
    /// # fn main() -> anyhow::Result<()> {
    /// # use simulator::signals::{SignalPath, SignalError};
    /// # use simulator::sim::InteractiveSimulator;
    /// # let mut sim: InteractiveSimulator = todo!();
    /// let a_sg: SignalPath = "a".parse()?;
    /// sim.poke_signal(&a_sg, 0x55u64)?;
    /// let opcode_sg: SignalPath = "top.cpu.alu.opcode".parse()?;
    /// sim.poke_signal(&opcode_sg, true)?;
    /// let r0_sg: SignalPath = "top.cpu.registers.r0[0]".parse()?;
    /// sim.poke_signal(&r0_sg, false)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn poke_signal<V>(
        &mut self,
        signal_path: &SignalPath,
        value: V,
    ) -> Result<(), SimulatorError>
    where
        V: PokeValue,
    {
        let value = value.to_u64();
        let pins = self.signal_path_to_aigpins(signal_path)?;
        if pins.len() == 1 && pins[0].1 == 0 {
            // Scalar case or single bit select
            self.poke(pins[0].0 >> 1, value != 0);
        } else {
            // Bus case
            // The pins were sorted by index in signal_path_to_aigpins.
            // Bit 0 of value goes to the pin with the lowest index.
            for (i, (aigpin_iv, _)) in pins.iter().enumerate() {
                if i < 64 {
                    let bit_val = (value >> i) & 1 != 0;
                    self.poke(aigpin_iv >> 1, bit_val);
                }
            }
        }
        Ok(())
    }

    /// Read an output signal value by name (call after step()).
    ///
    /// Accepts a `&str` dotted path or a pre-built [`SignalPath`]. Returns a
    /// [`PeekResult`] that can be compared directly with `bool` or `u64`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use simulator::sim::SimulatorError;
    /// # fn main() -> anyhow::Result<()> {
    /// # use simulator::signals::SignalPath;
    /// # use simulator::sim::InteractiveSimulator;
    /// # let mut sim: InteractiveSimulator = todo!();
    /// let path: SignalPath = "sum".parse()?;
    /// let sum = sim.peek_signal(&path)?;
    /// let result_path = "top.cpu.alu.result[3]".parse()?;
    /// let bit = sim.peek_signal(&result_path)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn peek_signal(
        &mut self,
        signal_path: &SignalPath,
    ) -> Result<PeekResult, SimulatorError>
where {
        let pins = self.signal_path_to_aigpins(signal_path)?;
        let mut result = 0u64;
        for (i, (aigpin_iv, _)) in pins.iter().enumerate() {
            if i < 64 && self.peek(*aigpin_iv) {
                result |= 1 << i;
            }
        }
        Ok(PeekResult(result))
    }

    // Fast API for typed wrappers

    /// Resolve an input AIG pin to a state offset for fast access
    pub fn resolve_input_offset(&self, aig_pin: usize) -> Option<StateOffset> {
        self.input_map.get(&aig_pin).map(|&o| StateOffset::from(o))
    }

    /// Resolve an output AIG pin to a state offset for fast access
    pub fn resolve_output_offset(&self, aig_pin: usize) -> Option<StateOffset> {
        self.output_map.get(&aig_pin).map(|&o| StateOffset::from(o))
    }

    /// Write a single bit into a state slice at the given offset.
    fn write_bit(state_slice: &mut [u32], offset: StateOffset, value: bool) {
        if value {
            state_slice[offset.word_idx()] |= 1u32 << offset.bit_idx();
        } else {
            state_slice[offset.word_idx()] &= !(1u32 << offset.bit_idx());
        }
    }

    /// Read a single bit from a state slice at the given offset.
    fn read_bit(state_slice: &[u32], offset: StateOffset) -> bool {
        (state_slice[offset.word_idx()] >> offset.bit_idx()) & 1 != 0
    }

    /// Set an input signal value using pre-resolved offset (fast path)
    ///
    /// Does not sync to GPU immediately; the transfer is batched and performed
    /// at the start of the next `step()` call.
    pub fn poke_offset(&mut self, offset: StateOffset, value: bool) {
        Self::write_bit(self.current_state.as_mut(), offset, value);
    }

    /// Set multiple input bits using pre-resolved offsets (fast path)
    ///
    /// Does not sync to GPU immediately; the transfer is batched and performed
    /// at the start of the next `step()` call.
    pub fn poke_offsets(&mut self, offsets: &[StateOffset], value: u64) {
        let state_slice = self.current_state.as_mut();
        for (i, &offset) in offsets.iter().enumerate().take(64) {
            Self::write_bit(state_slice, offset, (value >> i) & 1 != 0);
        }
    }

    /// Read an output signal value using pre-resolved offset (fast path)
    pub fn peek_offset(&self, offset: StateOffset) -> bool {
        Self::read_bit(self.current_state.as_ref(), offset)
    }

    /// Read multiple output bits using pre-resolved offsets (fast path)
    pub fn peek_offsets(&self, offsets: &[StateOffset]) -> u64 {
        let state_slice = self.current_state.as_ref();
        offsets
            .iter()
            .take(64)
            .enumerate()
            .fold(0u64, |acc, (i, &o)| {
                acc | (Self::read_bit(state_slice, o) as u64) << i
            })
    }

    /// Resolve a signal path to state offsets for fast repeated access.
    ///
    /// Accepts a `&str` dotted path or a pre-built [`SignalPath`]. Returns
    /// `None` if the signal is not found or any bit has no state offset.
    /// Use the returned [`StateOffset`]s with
    /// [`poke_offset`](Self::poke_offset) / [`peek_offset`](Self::peek_offset)
    /// to avoid repeated name lookups.
    pub fn resolve_signal_offsets(
        &self,
        signal_path: &SignalPath,
    ) -> Option<Vec<StateOffset>>
where {
        let pins = self.signal_path_to_aigpins(signal_path).ok()?;
        let mut offsets = Vec::new();
        for (pin_iv, _) in pins {
            // Check input map (key is pin_iv >> 1)
            if let Some(&off) = self.input_map.get(&(pin_iv >> 1)) {
                offsets.push(StateOffset::from(off));
            }
            // Check output map (key is pin_iv)
            else if let Some(&off) = self.output_map.get(&pin_iv) {
                offsets.push(StateOffset::from(off));
            } else {
                return None;
            }
        }
        Some(offsets)
    }

    /// Get clock edge flag positions for a clock signal.
    ///
    /// Accepts a `&str` dotted path or a pre-built [`SignalPath`]. Returns
    /// `(posedge_word, posedge_mask, negedge_word, negedge_mask)`; word
    /// positions are `u32::MAX` if the corresponding edge flag does not exist.
    /// Returns `None` if the signal is not a clock or is not found.
    pub fn get_clock_edge_flags(
        &self,
        signal_path: &SignalPath,
    ) -> Option<(u32, u32, u32, u32)> {
        let pins = self.signal_path_to_aigpins(signal_path).ok()?;
        if pins.is_empty() {
            return None;
        }

        // Get the pin_id from the first bit's aigpin_iv
        let aigpin_iv = pins[0].0;
        let aigpin = aigpin_iv >> 1;

        // Find the netlist pin that maps to this aigpin
        for (&pin_id, &(pe, ne)) in &self.aig.clock_pin2aigpins {
            let pin_aigpin = self.aig.pin2aigpin_iv[pin_id] >> 1;
            if pin_aigpin == aigpin {
                // Found the clock - get the edge flag positions
                let (pe_word, pe_mask) = if pe != usize::MAX {
                    if let Some(&p) = self.input_map.get(&pe) {
                        (p >> 5, 1u32 << (p & 31))
                    } else {
                        (u32::MAX, 0)
                    }
                } else {
                    (u32::MAX, 0)
                };

                let (ne_word, ne_mask) = if ne != usize::MAX {
                    if let Some(&p) = self.input_map.get(&ne) {
                        (p >> 5, 1u32 << (p & 31))
                    } else {
                        (u32::MAX, 0)
                    }
                } else {
                    (u32::MAX, 0)
                };

                return Some((pe_word, pe_mask, ne_word, ne_mask));
            }
        }

        // Not a clock signal - no edge flags
        None
    }

    /// Create a toggle watch for a clock signal with automatic edge detection.
    ///
    /// This is a convenience method that combines `resolve_signal_offsets` and
    /// `get_clock_edge_flags` to create a properly configured toggle watch.
    ///
    /// Returns `None` if the signal can't be resolved.
    fn clock_toggle_watch(
        &self,
        signal_name: &str,
        freq: u32,
        start_offset: u32,
    ) -> Option<WatchCondition> {
        let signal_path: SignalPath = signal_name.parse().ok()?;
        let offsets = self.resolve_signal_offsets(&signal_path)?;
        assert_eq!(offsets.len(), 1, "clock signal must be single-bit");

        let offset = offsets[0];
        let clock_word = offset.word_idx() as u32;
        let clock_mask = 1u32 << offset.bit_idx();

        let (pe_word, pe_mask, ne_word, ne_mask) = self
            .get_clock_edge_flags(&signal_path)
            .unwrap_or((u32::MAX, 0, u32::MAX, 0));

        let edge = ClockEdgeInfo {
            clock_word,
            clock_mask,
            posedge_word: pe_word,
            posedge_mask: pe_mask,
            negedge_word: ne_word,
            negedge_mask: ne_mask,
        };

        Some(WatchCondition::new_toggle(edge, freq, start_offset))
    }

    /// Run multiple cycles with clock toggling handled by the kernel.
    ///
    /// This is a convenience wrapper around `step()` that creates toggle
    /// watches for the specified clocks with automatic edge detection.
    ///
    /// # Returns
    ///
    /// Number of cycles actually executed (may be less if a stop watch
    /// triggers)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use simulator::sim::{InteractiveSimulator, ToggleClock};
    /// # let mut sim: InteractiveSimulator = todo!();
    /// // Run 100 cycles, toggling clk_i every cycle and ref_clk_i every 5 cycles
    /// let cycles = sim.batched_step(
    ///     100,
    ///     &[
    ///         ToggleClock {
    ///             signal: "clk_i",
    ///             freq: 1,
    ///             phase: 0,
    ///         },
    ///         ToggleClock {
    ///             signal: "ref_clk_i",
    ///             freq: 5,
    ///             phase: 0,
    ///         },
    ///     ],
    ///     None,
    /// );
    /// ```
    pub fn batched_step(
        &mut self,
        cycles: usize,
        clocks: &[ToggleClock],
        watches: Option<Vec<WatchCondition>>,
    ) -> u32 {
        let mut all_watches: Vec<_> = clocks
            .iter()
            .filter_map(|c| self.clock_toggle_watch(c.signal, c.freq, c.phase))
            .collect();

        if let Some(extra) = watches {
            all_watches.extend(extra);
        }

        self.run(
            Some(cycles),
            #[cfg(feature = "cuda")]
            Some(all_watches),
        )
    }

    /// Get list of input pins (for diagnostics)
    pub fn list_inputs(&self) -> Vec<usize> {
        self.input_map.keys().copied().collect()
    }

    /// Get list of output pins (for diagnostics)
    pub fn list_outputs(&self) -> Vec<usize> {
        self.output_map.keys().copied().collect()
    }

    /// Get the number of input pins
    pub fn num_inputs(&self) -> usize {
        self.input_map.len()
    }

    /// Get the number of output pins
    pub fn num_outputs(&self) -> usize {
        self.output_map.len()
    }

    /// Get state size in u32 words
    pub fn state_size(&self) -> usize {
        self.state_size
    }

    /// Get SRAM size in u32 words
    pub fn sram_size(&self) -> usize {
        self.sram_storage.len()
    }

    /// Get mutable access to SRAM storage
    ///
    /// Returns a mutable slice of the SRAM storage for direct memory
    /// manipulation. This is useful for pre-loading programs or data before
    /// simulation.
    ///
    /// After modifying SRAM contents, call `sram_sync()` to ensure the data
    /// is transferred to GPU memory (if using CUDA backend).
    ///
    /// # Example
    /// ```no_run
    /// # use simulator::sim::InteractiveSimulator;
    /// # let mut sim: InteractiveSimulator = todo!();
    /// // Load a program into SRAM
    /// let sram = sim.sram_mut();
    /// sram[0] = 0x00000013; // NOP instruction
    /// sram[1] = 0x00000013; // NOP instruction
    /// sim.sram_sync();
    /// ```
    pub fn sram_mut(&mut self) -> &mut [u32] {
        self.sram_storage.as_mut()
    }

    /// Get read-only access to SRAM storage
    ///
    /// Returns a slice of the SRAM storage for reading memory contents.
    pub fn sram_ref(&self) -> &[u32] {
        self.sram_storage.as_ref()
    }

    /// Synchronize SRAM storage to GPU memory
    ///
    /// Call this after modifying SRAM contents via `sram_mut()` to ensure
    /// the changes are visible to the GPU simulation kernel.
    ///
    /// For CPU backend, this is a no-op.
    pub fn sram_sync(&mut self) {
        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::CUDA(_)) {
            self.sram_storage.as_mut_uptr(self.device);
        }
    }

    /// Look up a traceable signal by name. Returns `None` if the signal is
    /// not observable (not a primary port, keep-net, or registered DFF/SRAM
    /// output).
    pub fn wire(&self, name: &str) -> Option<WireHandle> {
        self.tracer.wire(name)
    }

    /// Returns the canonical display name of a previously resolved wire.
    pub fn wire_name(&self, handle: WireHandle) -> &str {
        self.tracer.wire_name(handle)
    }

    /// Handles for every signal marked `(* keep *)` in the RTL source.
    pub fn keep_signals(&self) -> &[WireHandle] {
        self.tracer.keep_signals()
    }

    /// Enable VCD recording for the given wires. Each subsequent `step()`
    /// appends any transitions on these signals to an in-memory trace; call
    /// [`write_vcd`](Self::write_vcd) at the end of simulation to emit it.
    pub fn trace(&mut self, handles: &[WireHandle]) {
        self.tracer.trace(handles);
    }

    /// Write the recorded VCD trace to `writer` under a single module scope.
    pub fn write_vcd<W: std::io::Write>(
        &self,
        writer: W,
        module_name: &str,
    ) -> std::io::Result<()> {
        self.tracer.write_vcd(writer, module_name)
    }

    #[allow(dead_code)]
    /// Debug helper to print state
    fn debug_print_state(&self, label: &str, use_next: bool) {
        tracing::debug!("===== {} =====", label);
        let state = if use_next {
            self.next_state.as_ref()
        } else {
            self.current_state.as_ref()
        };

        tracing::debug!("state[0] = {:032b}", state[0]);
        if state.len() > 1 && state[1] != 0 {
            tracing::debug!("state[1] = {:032b}", state[1]);
        }

        // Print all (remaining) non-zero words to help debug
        for (i, &word) in state.iter().enumerate().skip(2) {
            if word != 0 {
                tracing::debug!("state[{i}] = {word:032b} (0x{word:08x})");
            }
        }

        // Print input/output mappings
        tracing::debug!(
            "-------------- input map --------------------------------"
        );
        for (&aigpin, &offset) in &self.input_map {
            let offset = StateOffset::from(offset);
            let value = (state[offset.word_idx()] >> offset.bit_idx()) & 1;
            tracing::debug!(
                "\tinput aigpin {aigpin} @ offset {offset} = {value}"
            );
        }

        tracing::debug!(
            "-------------- output map -------------------------------"
        );
        for (&aigpin, &offset_iv) in &self.output_map {
            let offset = StateOffset::from(offset_iv);
            let word_idx = offset.word_idx();
            let bit_idx = offset.bit_idx();
            if word_idx < state.len() {
                let value = (state[word_idx] >> bit_idx) & 1;
                tracing::debug!(
                    "\toutput aigpin {aigpin} @ offset {offset} = {value} (state[{word_idx}]={:032b})",
                    state[word_idx]
                );
            } else {
                tracing::debug!(
                    "\toutput aigpin {aigpin} @ offset {offset} - OUT OF BOUNDS! state.len()={}",
                    state.len()
                );
            }
        }
        tracing::debug!(
            "--------------------------------------------------------"
        );
    }
}

/// Builder for creating a simulator from design files
///
/// This handles the loading and compilation pipeline,
/// similar to how Verilator compiles Verilog to C++
pub struct SimulatorBuilder {
    netlist_verilog: std::path::PathBuf,
    top_module: Option<String>,
    gemparts: std::path::PathBuf,
    num_blocks: usize,
    device: Device,
    dpi_callbacks:
        IndexMap<String, Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>>,
    dpi_modes: IndexMap<String, DpiMode>,
    dpi_libs: Vec<libloading::Library>,
    /// use pipes to run DPI every cycles calls (enables batched step)
    #[cfg(feature = "cuda")]
    piped_dpi_calls: bool,
}

impl Default for SimulatorBuilder {
    fn default() -> Self {
        Self {
            netlist_verilog: Default::default(),
            top_module: None,
            gemparts: Default::default(),
            num_blocks: Default::default(),
            #[cfg(feature = "cuda")]
            device: Device::CUDA(0),
            #[cfg(not(feature = "cuda"))]
            device: Device::CPU,
            dpi_callbacks: IndexMap::new(),
            dpi_modes: IndexMap::new(),
            dpi_libs: Vec::new(),
            #[cfg(feature = "cuda")]
            piped_dpi_calls: false,
        }
    }
}

impl SimulatorBuilder {
    /// Create a new simulator builder
    pub fn new(
        netlist_verilog: std::path::PathBuf,
        gemparts: std::path::PathBuf,
        num_blocks: usize,
    ) -> Self {
        Self {
            netlist_verilog,
            gemparts,
            num_blocks,
            ..Self::default()
        }
    }

    /// Set the top module name
    pub fn top_module(mut self, name: String) -> Self {
        self.top_module = Some(name);
        self
    }

    /// Set the device
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Use CPU backend (for debugging)
    pub fn cpu(mut self) -> Self {
        self.device = Device::CPU;
        self
    }

    /// Register a DPI callback for a `import "DPI-C"` function.
    ///
    /// `name` must match the function name declared in the synthesized SV
    /// (i.e. the part after `$__DPI__` in the cell type, before any `$I`
    /// width encoding).
    ///
    /// The callback receives a flat slice of `u32` words and must return a
    /// `Vec<u32>`. Each port is packed into `ceil(width / 32)` consecutive
    /// little-endian words, so a 32-bit port = 1 word (backward compatible),
    /// a 64-bit port = 2 words, an 8-bit port = 1 word (upper bits zero).
    pub fn register_dpi(
        mut self,
        name: &str,
        cb: impl Fn(&[u32]) -> Vec<u32> + Send + Sync + 'static,
    ) -> Self {
        self.dpi_callbacks.insert(name.to_string(), Arc::new(cb));
        self
    }

    /// Register a DPI callback with an explicit [`DpiMode`].
    ///
    /// Behaves like [`register_dpi`](Self::register_dpi) but also records a
    /// mode that controls *when* the callback is invoked:
    ///
    /// - [`DpiMode::EveryCycle`] -- before every simulation cycle (this is what
    ///   [`register_dpi`](Self::register_dpi) uses implicitly).
    /// - [`DpiMode::Once`] -- once per [`InteractiveSimulator::reset`], before
    ///   state is uploaded to the GPU.
    pub fn register_dpi_with_mode(
        mut self,
        name: &str,
        mode: DpiMode,
        cb: impl Fn(&[u32]) -> Vec<u32> + Send + Sync + 'static,
    ) -> Self {
        self.dpi_callbacks.insert(name.to_string(), Arc::new(cb));
        self.dpi_modes.insert(name.to_string(), mode);
        self
    }

    /// Load a shared library (`.so` / `.dylib`) containing DPI-C functions.
    ///
    /// The library is searched for symbols named `harness_dpi_<func>` for every
    /// DPI function declared in the design that does not already have a Rust
    /// callback registered via [`Self::register_dpi`].
    ///
    /// Each function must have the following signature:
    ///
    /// ```c
    /// void harness_dpi_<name>(const uint32_t* inputs,  size_t n_in_words,
    ///                        uint32_t*       outputs, size_t n_out_words);
    /// ```
    ///
    /// Ports are packed in declaration order, each occupying
    /// `ceil(width / 32)` consecutive little-endian `uint32_t` words.
    /// For example, a 32-bit port is 1 word, a 64-bit port is 2 words,
    /// and an 8-bit port is 1 word (upper bits zero).
    pub fn register_dpi_lib(
        mut self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        // SAFETY: the loaded library is kept alive in `dpi_libs` and later
        // moved into `DpiState::libs`, so function pointers obtained from it
        // remain valid for the simulator's lifetime.
        let lib = unsafe { libloading::Library::new(path.as_ref()) }
            .with_context(|| {
                format!(
                    "failed to load DPI shared library: {}",
                    path.as_ref().display()
                )
            })?;
        self.dpi_libs.push(lib);
        Ok(self)
    }

    /// Build the simulator.
    ///
    /// Loads the gate-level netlist, deserialises the companion `.gemparts`
    /// and `.stageds` artifacts produced by `compile`, and resolves
    /// every DPI callsite in the design.
    ///
    /// Returns an error if any DPI function declared in the design has no
    /// callback registered (via [`register_dpi`](Self::register_dpi),
    /// [`register_dpi_with_mode`](Self::register_dpi_with_mode), or a shared
    /// library loaded with [`register_dpi_lib`](Self::register_dpi_lib)).
    pub fn build(self) -> anyhow::Result<InteractiveSimulator> {
        use aig::aigpdk::AIGPDKLeafPins;
        use flatten::pe::Partition;
        use netlistdb::NetlistDB;

        // Load the netlist
        let netlistdb = NetlistDB::from_sverilog_file(
            &self.netlist_verilog,
            self.top_module.as_deref(),
            &AIGPDKLeafPins(),
        )
        .context("build netlist from verilog file")?;

        let netlistdb = Arc::new(netlistdb);
        let aig = Arc::new(AIG::from_netlistdb(&netlistdb));

        let mut stageds_out = self.gemparts.clone();
        stageds_out.set_extension("stageds");
        let f = std::fs::File::open(&stageds_out)?;
        let mut buf = std::io::BufReader::new(f);
        let stageds: Vec<(usize, usize, StagedAIG)> =
            serde_bare::from_reader(&mut buf)?;

        let f = std::fs::File::open(&self.gemparts)?;
        let mut buf = std::io::BufReader::new(f);
        let parts_in_stages: Vec<Vec<Partition>> =
            serde_bare::from_reader(&mut buf)?;

        let mut input_layout = Vec::new();
        for (i, driv) in aig.drivers.iter().enumerate() {
            if let DriverType::InputPort(_)
            | DriverType::InputClockFlag(_, _)
            | DriverType::DPIOutput(_) = driv
            {
                input_layout.push(i);
            }
        }

        // Create the compiled GEM script
        let script: FlattenedScriptV1 = FlattenedScriptV1::from(
            &aig,
            &stageds
                .iter()
                .map(|(_, _, staged)| staged)
                .collect::<Vec<_>>(),
            &parts_in_stages
                .iter()
                .map(|ps| ps.as_slice())
                .collect::<Vec<_>>(),
            self.num_blocks,
            input_layout,
        );

        // Resolve DPI call info: classify each input bit at build time.
        let dpi_calls: Vec<DpiCallInfo> = aig
            .dpi_calls
            .values()
            .map(|block| {
                let input_bits: Vec<Vec<dpi::DpiInputBit>> = block
                    .inputs_iv
                    .iter()
                    .enumerate()
                    .map(|(port, bits)| {
                        bits.iter()
                            .enumerate()
                            .map(|(bit, &iv)| {
                                dpi::resolve_dpi_input(
                                    iv,
                                    &script,
                                    &aig.drivers,
                                    &block.func_name,
                                )
                                .map_err(|e| {
                                    anyhow::anyhow!("IN_{port}[{bit}]: {e}")
                                })
                            })
                            .collect()
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let output_bit_offsets =
                    dpi::resolve_ports(&block.outputs, |aigpin| {
                        dpi::resolve_dpi_output_pin(aigpin, &script)
                    })?;

                Ok(DpiCallInfo {
                    func_name: block.func_name.clone(),
                    mode: self
                        .dpi_modes
                        .get(&block.func_name)
                        .copied()
                        .unwrap_or_default(),
                    input_bits,
                    output_bit_offsets,
                    input_widths: block.input_widths.clone(),
                    output_widths: block.output_widths.clone(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // For each DPI function without a Rust callback, try to find a
        // `harness_dpi_<name>` symbol in the loaded shared libraries.
        let mut dpi_callbacks = self.dpi_callbacks;
        for call in &dpi_calls {
            if dpi_callbacks.contains_key(&call.func_name) {
                continue;
            }
            let n_out_words: usize =
                call.output_widths.iter().map(|w| w.div_ceil(32)).sum();
            if let Some(cb) = dpi::load_dpi_symbol(
                &self.dpi_libs,
                &call.func_name,
                n_out_words,
            )? {
                dpi_callbacks.insert(call.func_name.clone(), cb);
            }
            // TODO push None to missing_callbacks instead of second pass?
        }

        let missing_callbacks: Vec<String> = dpi_calls
            .iter()
            .filter_map(|call| {
                (!dpi_callbacks.contains_key(&call.func_name))
                    .then_some(call.func_name.clone())
            })
            .collect();

        // Fail if any DPI function has no callback registered.
        if !missing_callbacks.is_empty() {
            anyhow::bail!(
                "DPI functions '{:?}' have no callback (register via .register_dpi() or .register_dpi_lib())",
                missing_callbacks,
            );
        }

        let dpi = DpiState::new(dpi_calls, dpi_callbacks, self.dpi_libs);

        #[cfg(feature = "cuda")]
        let dpi_gpu_pipe = if dpi.has_every_cycle() && self.piped_dpi_calls {
            GpuDpiPipe::from_tables(dpi.build_gpu_pipe_tables()?, self.device)?
        } else {
            GpuDpiPipe::empty()
        };

        Ok(InteractiveSimulator::new(
            script,
            self.num_blocks,
            self.device,
            netlistdb,
            aig,
            dpi,
            #[cfg(feature = "cuda")]
            dpi_gpu_pipe,
        ))
    }
}
