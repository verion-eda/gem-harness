// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! DPI (Direct Programming Interface) support for the simulator.
//!
//! Groups all DPI-related types, resolution logic, and runtime state into a
//! single module.

use std::sync::Arc;

use aig::DriverType;
use flatten::FlattenedScriptV1;
#[cfg(feature = "cuda")]
use gem_cuda::ucci;
use indexmap::IndexMap;
use itertools::{Either, Itertools};

use crate::offset;

/// Controls when a DPI callback is invoked relative to simulation cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DpiMode {
    /// Callback runs before every simulation cycle (default).
    #[default]
    EveryCycle,
    /// Callback runs once per `reset()`, before state is uploaded to GPU.
    Once,
}

/// Classifies DPI input bit at build time.
///
/// Handles constants, direct state-vector reads, and small combinational
/// expressions (AND gates from reset gating, etc.) without needing the
/// full AIG at runtime.
#[derive(Debug, Clone)]
pub(crate) enum DpiInputBit {
    /// Constant 0 or 1 (with inversion already applied).
    Constant(u32),
    /// Direct state-vector bit offset, with optional inversion.
    StateOffset {
        offset: offset::StateOffset,
        invert: bool,
    },
    /// AND of two sub-expressions, with optional output inversion.
    /// Handles combinational DPI inputs (e.g. reset synchronizer gating).
    And {
        a: Box<DpiInputBit>,
        b: Box<DpiInputBit>,
        invert: bool,
    },
}

impl DpiInputBit {
    /// Evaluate the expression build by resolve_dpi_input at runtime.
    #[inline]
    pub(crate) fn eval(&self, state: &[u32]) -> u32 {
        match self {
            DpiInputBit::Constant(v) => *v,
            DpiInputBit::StateOffset {
                offset: off,
                invert,
            } => {
                let raw = off.read(state);
                raw ^ (*invert as u32)
            }
            DpiInputBit::And { a, b, invert } => {
                let val = a.eval(state) & b.eval(state);
                val ^ (*invert as u32)
            }
        }
    }
}

/// Resolved DPI call info: state-vector offsets for one DPI function instance.
///
/// Created during `SimulatorBuilder::build()` by mapping AIG pins to
/// state-vector bit positions.
pub(crate) struct DpiCallInfo {
    pub func_name: String,
    pub mode: DpiMode,
    /// `input_bits[port][bit]`: resolved input bit for IN port `port` bit
    /// `bit`.
    pub input_bits: Vec<Vec<DpiInputBit>>,
    /// `output_bit_offsets[port][bit]`: state bit offset to write in
    /// `current_state` for OUT port `port` bit `bit`.
    pub output_bit_offsets: Vec<Vec<u32>>,
    /// Per-input-port bit widths (for multi-word callback packing).
    pub input_widths: Vec<usize>,
    /// Per-output-port bit widths (for multi-word callback packing).
    pub output_widths: Vec<usize>,
}

/// Runtime DPI state for an `InteractiveSimulator`.
///
/// Groups all DPI-related fields that were previously spread across the
/// simulator struct.
pub(crate) struct DpiState {
    /// Resolved call info for every DPI function in the design.
    pub calls: Vec<DpiCallInfo>,
    /// Registered callbacks keyed by function name.
    pub callbacks:
        IndexMap<String, Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>>,
    /// Indices into `calls` for [`DpiMode::Once`] functions.
    pub once_indices: Vec<usize>,
    /// Indices into `calls` for [`DpiMode::EveryCycle`] functions.
    pub every_cycle_indices: Vec<usize>,
    /// Keep loaded DPI shared libraries alive for the lifetime of the
    /// simulator. Structurally immutable after creation.
    /// The field is never read directly — it exists to prevent the libraries
    /// from being dropped while function pointers from them are in use.
    #[allow(dead_code)]
    pub libs: Box<[libloading::Library]>,
}

impl DpiState {
    /// Build `DpiState` from resolved calls, callbacks, and loaded libraries.
    pub fn new(
        calls: Vec<DpiCallInfo>,
        callbacks: IndexMap<
            String,
            Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>,
        >,
        libs: Vec<libloading::Library>,
    ) -> Self {
        let (once_indices, every_cycle_indices): (Vec<usize>, Vec<usize>) =
            calls
                .iter()
                .enumerate()
                .partition_map(|(i, c)| match c.mode {
                    DpiMode::Once => Either::Left(i),
                    DpiMode::EveryCycle => Either::Right(i),
                });

        Self {
            calls,
            callbacks,
            once_indices,
            every_cycle_indices,
            libs: libs.into_boxed_slice(),
        }
    }

    /// Run registered DPI callbacks for the given indices on `state`.
    ///
    /// For each DPI call, assembles input port bits from `state`,
    /// invokes the registered callback, and scatters output bits back.
    pub fn execute_by_indices(&self, indices: &[usize], state: &mut [u32]) {
        for &idx in indices {
            let call = &self.calls[idx];
            if let Some(cb) = self.callbacks.get(&call.func_name) {
                // Assemble inputs: each port packs into ceil(width/32)
                // consecutive LE u32 words.
                let mut inputs: Vec<u32> = Vec::new();
                for (port_idx, port_bits) in call.input_bits.iter().enumerate()
                {
                    let width = call.input_widths[port_idx];
                    let num_words = width.div_ceil(32);
                    let base = inputs.len();
                    inputs.resize(base + num_words, 0u32);
                    for (bit, input_bit) in port_bits.iter().enumerate() {
                        let state_bit = input_bit.eval(state);
                        inputs[base + bit / 32] |= state_bit << (bit % 32);
                    }
                }
                let outputs = cb(&inputs);
                // Scatter outputs: each port occupies ceil(width/32)
                // consecutive LE u32 words.
                let mut word_idx = 0;
                for (port_idx, port_bits) in
                    call.output_bit_offsets.iter().enumerate()
                {
                    let width = call.output_widths[port_idx];
                    let num_words = width.div_ceil(32);
                    for (bit, &off) in port_bits.iter().enumerate() {
                        let val_bit =
                            (outputs[word_idx + bit / 32] >> (bit % 32)) & 1;
                        let word = &mut state[(off >> 5) as usize];
                        let bit_in_word = off & 31;
                        *word = (*word & !(1 << bit_in_word))
                            | (val_bit << bit_in_word);
                    }
                    word_idx += num_words;
                }
            }
        }
    }

    /// Execute all every-cycle DPI callbacks on `state`.
    pub fn execute_every_cycle(&self, state: &mut [u32]) {
        self.execute_by_indices(&self.every_cycle_indices, state);
    }

    /// Execute once-per-reset DPI callbacks on `state`.
    pub fn execute_once(&self, state: &mut [u32]) {
        self.execute_by_indices(&self.once_indices, state);
    }

    /// Returns `true` if there are any every-cycle DPI callbacks.
    #[cfg(feature = "cuda")]
    pub fn has_every_cycle(&self) -> bool {
        !self.every_cycle_indices.is_empty()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn build_gpu_pipe_tables(
        &self,
    ) -> anyhow::Result<crate::pipe::GpuPipeTables> {
        let mut in_offsets: Vec<u32> = Vec::new();
        let mut in_port_widths: Vec<u32> = Vec::new();
        let mut out_offsets: Vec<u32> = Vec::new();
        let mut out_port_widths: Vec<u32> = Vec::new();
        let mut call_descs: Vec<ucci::DpiCallDesc> = Vec::new();
        let mut slot_infos: Vec<crate::pipe::DpiPipeSlotInfo> = Vec::new();
        let mut pipe_total_words: usize = 0;
        let mut cpu_only_indices: Vec<usize> = Vec::new();

        'next_call: for &idx in &self.every_cycle_indices {
            let call = &self.calls[idx];

            let in_offsets_start = in_offsets.len() as u32;
            let in_ports_start = in_port_widths.len() as u32;
            let n_in_words: usize =
                call.input_widths.iter().map(|w| w.div_ceil(32)).sum();
            let mut in_invert_mask = vec![0u32; n_in_words];
            let mut in_const_mask = vec![0u32; n_in_words];
            let mut in_const_ones = vec![0u32; n_in_words];
            let mut word_offset: usize = 0;
            for (port_idx, port_bits) in call.input_bits.iter().enumerate() {
                for (bit_in_port, bit) in port_bits.iter().enumerate() {
                    let word_idx = word_offset + bit_in_port / 32;
                    let bit_in_word = bit_in_port % 32;
                    match bit {
                        DpiInputBit::StateOffset { offset, invert } => {
                            in_offsets.push(offset.into());
                            if *invert {
                                in_invert_mask[word_idx] |= 1 << bit_in_word;
                            }
                        }
                        DpiInputBit::Constant(value) => {
                            // use a dummy offset - the GPU value will be
                            // overridden by the poll thread using
                            // in_const_mask.
                            in_offsets.push(0);
                            in_const_mask[word_idx] |= 1 << bit_in_word;
                            if *value != 0 {
                                in_const_ones[word_idx] |= 1 << bit_in_word;
                            }
                        }
                        DpiInputBit::And { .. } => {
                            // TODO for now we dont pipeline this and fall back
                            // to CPU for execution: revert partial buffers.
                            in_offsets.truncate(in_offsets_start as usize);
                            in_port_widths.truncate(in_ports_start as usize);
                            cpu_only_indices.push(idx);
                            continue 'next_call;
                        }
                    }
                }
                word_offset += call.input_widths[port_idx].div_ceil(32);
            }
            for &w in &call.input_widths {
                in_port_widths.push(w as u32);
            }

            let out_offsets_start = out_offsets.len() as u32;
            let out_ports_start = out_port_widths.len() as u32;
            // Each output port is also word-aligned.
            let n_out_words: usize =
                call.output_widths.iter().map(|w| w.div_ceil(32)).sum();
            for port_bits in &call.output_bit_offsets {
                out_offsets.extend_from_slice(port_bits);
            }
            for &w in &call.output_widths {
                out_port_widths.push(w as u32);
            }

            let pipe_slot_start = pipe_total_words as u32;
            // Slot layout: [status(1)] [n_in_words] [n_out_words]
            pipe_total_words += 1 + n_in_words + n_out_words;

            call_descs.push(ucci::DpiCallDesc {
                in_offsets_start,
                n_in_words: n_in_words as u32,
                n_in_ports: call.input_widths.len() as u32,
                in_ports_start,
                out_offsets_start,
                n_out_ports: call.output_widths.len() as u32,
                out_ports_start,
                pipe_slot_start,
            });

            slot_infos.push(crate::pipe::DpiPipeSlotInfo {
                pipe_slot_start: pipe_slot_start as usize,
                n_in_words,
                n_out_words,
                func_name: call.func_name.clone(),
                in_invert_mask,
                in_const_mask,
                in_const_ones,
            });
        }

        Ok(crate::pipe::GpuPipeTables {
            in_offsets,
            in_port_widths,
            out_offsets,
            out_port_widths,
            call_descs,
            slot_infos,
            pipe_total_words,
            cpu_only_indices,
        })
    }
}

/// Maximum AND-gate recursion depth for DPI input resolution.
/// AXI handshake signals can go through deep combinational cones
/// (arbitration, crossbar routing) producing long AND-inverter chains.
const DPI_AND_DEPTH_LIMIT: usize = 1024;

/// Resolve one DPI *input* IV to a [`DpiInputBit`] in the SimulationBuilder.
/// Creates an expression tree that is evaluated at runtime.
///
/// - `aigpin == 0` (Tie0): returns `Constant(invert_bit)`.
/// - Leaf in `input_map` or `output_map`: returns `StateOffset` with inversion
///   flag.
/// - AND gates: recursively resolves both children, building a small expression
///   tree evaluated at runtime.
/// - Missing pins: returns an error.
pub(crate) fn resolve_dpi_input(
    iv: usize,
    script: &FlattenedScriptV1,
    aig_drivers: &[DriverType],
    func_name: &str,
) -> anyhow::Result<DpiInputBit> {
    resolve_dpi_input_inner(iv, script, aig_drivers, func_name, 0)
}

fn resolve_dpi_input_inner(
    iv: usize,
    script: &FlattenedScriptV1,
    aig_drivers: &[DriverType],
    func_name: &str,
    depth: usize,
) -> anyhow::Result<DpiInputBit> {
    let aigpin = iv >> 1;
    let invert = (iv & 1) != 0;

    if aigpin == 0 {
        // AIG constant node: Tie0, possibly inverted to 1.
        return Ok(DpiInputBit::Constant(invert as u32));
    }

    // Check state-vector maps FIRST - an AND gate output may already be
    // in the state vector (e.g. DFF D-input cone that was scheduled as
    // a combinational output). This avoids needless recursion.
    if let Some(&off) = script.input_map.get(&aigpin) {
        return Ok(DpiInputBit::StateOffset {
            offset: off.into(),
            invert,
        });
    }
    if let Some(&off) = script.output_map.get(&iv) {
        return Ok(DpiInputBit::StateOffset {
            offset: off.into(),
            invert: false,
        });
    }

    // AND gate: recursively resolve both children.
    if let DriverType::AndGate(a_iv, b_iv) = aig_drivers[aigpin] {
        if depth >= DPI_AND_DEPTH_LIMIT {
            anyhow::bail!(
                "DPI input aigpin {aigpin} (iv={iv}) for '{func_name}' \
                 exceeds AND-gate depth limit ({DPI_AND_DEPTH_LIMIT})",
            );
        }
        let a = resolve_dpi_input_inner(
            a_iv,
            script,
            aig_drivers,
            func_name,
            depth + 1,
        )?;
        let b = resolve_dpi_input_inner(
            b_iv,
            script,
            aig_drivers,
            func_name,
            depth + 1,
        )?;
        return Ok(DpiInputBit::And {
            a: Box::new(a),
            b: Box::new(b),
            invert,
        });
    }

    anyhow::bail!(
        "DPI input aigpin {aigpin} (iv={iv}) for '{func_name}' \
         not found in input_map or output_map",
    );
}

/// Resolve one DPI *output* AIG pin to a state-vector bit offset.
pub(crate) fn resolve_dpi_output_pin(
    aigpin: usize,
    script: &FlattenedScriptV1,
) -> anyhow::Result<u32> {
    script.input_map.get(&aigpin).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "DPI output aigpin {aigpin} not found in input_map; \
             ensure DPIOutput is in input_layout",
        )
    })
}

/// Map a `&[Vec<pin>]` to `Vec<Vec<T>>` using the given resolver.
pub(crate) fn resolve_ports<T>(
    ports: &[Vec<usize>],
    mut resolve: impl FnMut(usize) -> anyhow::Result<T>,
) -> anyhow::Result<Vec<Vec<T>>> {
    ports
        .iter()
        .map(|bits| bits.iter().map(|&p| resolve(p)).collect())
        .collect()
}

/// Try to find a `harness_dpi_<name>` symbol in the loaded shared libraries
/// and wrap it in a boxed callback.
///
/// Fails if the symbol is found in more than one library.
pub(crate) fn load_dpi_symbol(
    libs: &[libloading::Library],
    func_name: &str,
    n_out_words: usize,
) -> anyhow::Result<Option<Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>>> {
    type DpiFn = unsafe extern "C" fn(*const u32, usize, *mut u32, usize);
    let sym_name = format!("harness_dpi_{func_name}");

    let mut found: Option<DpiFn> = None;
    for lib in libs {
        // SAFETY: symbol signature matches the register_dpi_lib doc comment;
        // library outlives the returned closure.
        let Ok(sym) = (unsafe { lib.get::<DpiFn>(sym_name.as_bytes()) }) else {
            continue;
        };
        if found.is_some() {
            anyhow::bail!(
                "DPI symbol '{sym_name}' found in multiple shared libraries; \
                 each symbol must be unique across all registered libraries",
            );
        }
        found = Some(*sym);
    }

    Ok(
        found.map(|fn_ptr| -> Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync> {
            Arc::new(move |inputs: &[u32]| {
                let mut outputs = vec![0u32; n_out_words];
                // SAFETY: `fn_ptr` is a valid symbol obtained via
                // `libloading::Library::get`, so it points to executable code
                // for as long as the owning `Library` is kept alive (caller's
                // responsibility).  The input pointer is valid for
                // `inputs.len()` reads (slice invariant) and the output pointer
                // is valid for `outputs.len()` writes (freshly allocated vec).
                // The two buffers do not overlap.  The function's ABI is
                // `extern "C"` with the exact signature `(const u32 *, usize,
                // u32 *, usize)` as required by the `register_dpi_lib`
                // contract.
                unsafe {
                    fn_ptr(
                        inputs.as_ptr(),
                        inputs.len(),
                        outputs.as_mut_ptr(),
                        outputs.len(),
                    );
                }
                outputs
            })
        }),
    )
}
