// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::{
    collections::{HashMap, HashSet},
    io,
};

use vcd_ng::{IdCode, Value, VarType};

use crate::offset::StateOffset;

/// Opaque handle to a traceable signal obtained from a [`Tracer`].
///
/// Handles can only be created via [`Tracer::wire`] or
/// [`Tracer::keep_signals`], ensuring they always refer to a valid signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WireHandle(pub(crate) usize);

struct IdGen(IdCode);

impl IdGen {
    fn new() -> Self {
        Self(IdCode::FIRST)
    }

    fn next(&mut self) -> IdCode {
        let id = self.0;
        self.0 = self.0.next();
        id
    }
}

pub(crate) struct WireInfo {
    #[allow(dead_code)]
    pub(crate) pin: crate::PinType,
    pub(crate) name: String,
    pub(crate) output_pos: StateOffset,
    pub(crate) id: IdCode,
}

/// Builds a [`Tracer`] by registering observable wires before simulation
/// starts.
pub(crate) struct TracerBuilder {
    wires: Vec<WireInfo>,
    wire_name_index: HashMap<String, usize>,
    id_gen: IdGen,
}

impl TracerBuilder {
    pub(crate) fn new() -> Self {
        Self {
            wires: Vec::new(),
            wire_name_index: HashMap::new(),
            id_gen: IdGen::new(),
        }
    }

    /// Add a wire. If a wire with the same name already exists it is skipped
    /// (first registration wins).
    pub(crate) fn add_wire(
        &mut self,
        pin: crate::PinType,
        name: String,
        output_pos: impl Into<StateOffset>,
    ) {
        if self.wire_name_index.contains_key(&name) {
            return;
        }
        self.wire_name_index.insert(name.clone(), self.wires.len());
        self.wires.push(WireInfo {
            pin,
            name,
            output_pos: output_pos.into(),
            id: self.id_gen.next(),
        });
    }

    /// Look up a wire by name.
    pub(crate) fn wire(&self, name: &str) -> Option<WireHandle> {
        self.wire_name_index.get(name).map(|&idx| WireHandle(idx))
    }

    pub(crate) fn build(self, keep_handles: Vec<WireHandle>) -> Tracer {
        Tracer {
            traceable_wires: self.wires,
            wire_name_index: self.wire_name_index,
            keep_handles,
            trace_handles: Vec::new(),
            trace_handle_set: HashSet::new(),
            traced_state: Vec::new(),
        }
    }
}

struct TraceSignalState {
    timestamp: u64,
    vid: IdCode,
    value: Value,
}

/// Records signal transitions during simulation and writes them as VCD.
pub struct Tracer {
    traceable_wires: Vec<WireInfo>,
    wire_name_index: HashMap<String, usize>,
    keep_handles: Vec<WireHandle>,
    trace_handles: Vec<WireHandle>,
    trace_handle_set: HashSet<usize>,
    traced_state: Vec<TraceSignalState>,
}

impl Tracer {
    /// Look up a wire by name. Returns `None` if the signal is not observable.
    pub fn wire(&self, name: &str) -> Option<WireHandle> {
        self.wire_name_index.get(name).map(|&idx| WireHandle(idx))
    }

    /// Returns the name of a wire handle.
    pub fn wire_name(&self, handle: WireHandle) -> &str {
        &self.traceable_wires[handle.0].name
    }

    /// Returns handles for all signals marked `(* keep *)` in the source.
    pub fn keep_signals(&self) -> &[WireHandle] {
        &self.keep_handles
    }

    /// Register wires to be recorded during simulation.
    pub fn trace(&mut self, handles: &[WireHandle]) {
        for &handle in handles {
            if self.trace_handle_set.insert(handle.0) {
                self.trace_handles.push(handle);
            }
        }
    }

    /// Write all recorded signal history as a self-contained VCD file.
    ///
    /// Handles the full VCD structure (timescale, module scope, variable
    /// definitions, and waveform data) in the correct order.
    pub fn write_vcd<W: io::Write>(
        &self,
        writer: W,
        module_name: &str,
    ) -> io::Result<()> {
        let mut writer = vcd_ng::Writer::new(writer);
        writer.timescale(1, vcd_ng::TimescaleUnit::US)?;
        writer.add_module(module_name)?;
        for &handle in &self.trace_handles {
            let info = &self.traceable_wires[handle.0];
            writer.var_def(VarType::Wire, 1, info.id, &info.name, None)?;
        }
        writer.upscope()?;
        writer.enddefinitions()?;
        writer.begin(vcd_ng::SimulationCommand::Dumpvars)?;
        for state in &self.traced_state {
            writer.timestamp(state.timestamp)?;
            writer.change_scalar(state.vid, state.value)?;
        }
        Ok(())
    }

    /// Record a simulation step. Called by the simulator after each step.
    pub(crate) fn record_step(
        &mut self,
        timestamp: u64,
        current_state: &[u32],
        next_state: &[u32],
    ) {
        for &handle in &self.trace_handles {
            let info = &self.traceable_wires[handle.0];
            let prev = info.output_pos.read(current_state);
            let new = info.output_pos.read(next_state);
            if prev == new {
                continue;
            }
            self.traced_state.push(TraceSignalState {
                timestamp,
                vid: info.id,
                value: if new == 1 { Value::V1 } else { Value::V0 },
            });
        }
    }
}
