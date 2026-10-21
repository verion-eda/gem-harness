// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! Partition flattening intermediate state.
//!
//! This module contains `FlatteningPart`, which holds temporary state while
//! converting a [`Partition`] (high-level description of AND gates in a
//! boomerang hierarchy, see [`Partition::build_one`]) into a GPU-executable
//! script.
//!
//! The flattening process for each partition has three phases:
//! 1. [`FlatteningPart::init_afters_writeouts`]: Analyze the partition
//!    structure, count SRAMs, and determine how many writeout slots are needed
//!    (including duplicates for multi-activation outputs).
//! 2. [`FlatteningPart::make_inputs_outputs`]: After state positions are
//!    allocated globally, set up the input/output mappings and build
//!    permutation tables for SRAM I/O and clock enables.
//! 3. [`FlatteningPart::build_script`]: Generate the final instruction stream
//!    that the GPU kernel reads.
//!
//! The "boomerang" algorithm is a hierarchical AND-gate reduction: 8192 inputs
//! are reduced through 13 stages of pairwise AND operations (14 levels total)
//! down to a single output (see [`BOOMERANG_NUM_STAGES`]).
//! Intermediate results can be "written out" at any level for use as DFF
//! inputs, primary outputs, or SRAM control signals.

use std::collections::BTreeMap;

use aig::{AIG, DriverType, EndpointGroup};
use indexmap::IndexMap;
use staging::StagedAIG;

use crate::{
    internal::{NUM_THREADS_V1, map_global_read_to_rounds},
    pe::{BOOMERANG_NUM_STAGES, Partition},
};

/// Sets or clears a single bit within a u32 at the given position.
fn set_bit_in_u32(v: &mut u32, pos: u32, bit: u8) {
    if bit != 0 {
        *v |= 1 << pos;
    } else {
        *v &= !(1 << pos);
    }
}

/// Intermediate state for flattening a single partition.
///
/// This struct accumulates information during the three flattening phases and
/// is discarded after `build_script` produces the final GPU instruction stream.
///
/// The struct tracks:
/// - Boomerang hierarchy layout (`afters`, `parts_after_writeouts`,
///   `after_writeout_pin2pos`)
/// - Output categorization by activation (`comb_outputs_activations`)
/// - Permutation tables for the GPU kernel (`sram_duplicate_*`, `clken_*`,
///   `data_inv`)
/// - Allocated positions in global state (`state_start`, `sram_start`)
#[derive(Debug, Clone, Default)]
pub(crate) struct FlatteningPart {
    /// for each boomerang stage, the result bits layout.
    afters: Vec<Vec<usize>>,
    /// for each partition, the output bits layout not containing sram outputs
    /// yet.
    parts_after_writeouts: Vec<usize>,
    /// mapping from aig pin index to writeout position (0~8192)
    after_writeout_pin2pos: IndexMap<usize, u16>,
    /// the number of SRAMs to simulate in this part.
    pub(crate) num_srams: u32,
    /// number of normal writeouts
    num_normal_writeouts: u32,
    /// number of writeout slots for output duplication
    num_duplicate_writeouts: u32,
    /// number of total writeouts
    pub(crate) num_writeouts: u32,
    /// the outputs categorized into activations
    comb_outputs_activations: IndexMap<usize, IndexMap<usize, Option<u16>>>,
    /// the current (placed) count of duplicate permutes
    cnt_placed_duplicate_permute: u32,

    /// the starting offset for FFs, outputs, and SRAM read results.
    pub(crate) state_start: u32,
    /// the starting offset of SRAM storage.
    pub(crate) sram_start: u32,

    /// the partial permutation instructions for
    /// 1. sram inputs
    /// 2. duplicated output pins due to difference in polarity/clock en.
    ///
    /// len: 8192
    sram_duplicate_permute: Vec<u16>,
    /// invert bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_inv: Vec<u32>,
    /// set-0 bit for sram_duplicate.
    ///
    /// len: 256
    sram_duplicate_set0: Vec<u32>,
    /// the permutation for clock enable pins.
    ///
    /// len: 8192
    clken_permute: Vec<u16>,
    /// invert bit for clken
    ///
    /// len: 256
    clken_inv: Vec<u32>,
    /// set-0 bit for clken
    ///
    /// len: 256
    clken_set0: Vec<u32>,
    /// invert bit for data corresponding to clken
    ///
    /// len: 256
    data_inv: Vec<u32>,
}

impl FlatteningPart {
    /// Initializes the "after" layout and writeout metadata for a partition.
    ///
    /// Called first during flattening (in `build_flattened_script_v1`) to
    /// preprocess each partition before allocating state positions.
    ///
    /// This function:
    /// 1. Builds `afters`: for each boomerang stage, the mapping from position
    ///    to AIG pin index after that stage completes. The layout is [dummy,
    ///    level13, level12, ..., level1] flattened.
    /// 2. Builds `parts_after_writeouts`: the AIG pins at writeout positions
    ///    (where results are extracted mid-computation), excluding SRAM
    ///    outputs.
    /// 3. Counts SRAMs and categorizes combinational outputs by their
    ///    "activation" (clock enable and data polarity) to determine how many
    ///    duplicate writeout slots are needed.
    /// 4. Computes `num_writeouts = num_normal_writeouts + num_srams +
    ///    num_duplicate_writeouts`.
    pub(crate) fn init_afters_writeouts(
        &mut self,
        aig: &AIG,
        staged: &StagedAIG,
        part: &Partition,
    ) {
        let afters = part
            .stages
            .iter()
            .map(|s| {
                let mut after = Vec::with_capacity(1 << BOOMERANG_NUM_STAGES);
                after.push(usize::MAX);
                for i in (1..=BOOMERANG_NUM_STAGES).rev() {
                    after.extend(s.hier[i].iter().copied());
                }
                after
            })
            .collect::<Vec<_>>();
        let wos = part
            .stages
            .iter()
            .zip(afters.iter())
            .flat_map(|(s, after)| {
                s.write_outs.iter().flat_map(|&woi| {
                    after[woi * 32..(woi + 1) * 32].iter().copied()
                })
            })
            .collect::<Vec<_>>();

        self.afters = afters;
        self.parts_after_writeouts = wos;
        self.num_normal_writeouts = part
            .stages
            .iter()
            .map(|s| s.write_outs.len())
            .sum::<usize>() as u32;
        self.num_srams = 0;

        // map: output aig pin id -> ((clken, data iv) -> pos)
        let mut comb_outputs_activations =
            IndexMap::<usize, IndexMap<usize, Option<u16>>>::new();
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(_) => {
                    self.num_srams += 1;
                }
                EndpointGroup::PrimaryOutput(idx) => {
                    comb_outputs_activations
                        .entry(idx >> 1)
                        .or_default()
                        .insert(2 | (idx & 1), None);
                }
                EndpointGroup::StagedIOPin(idx) => {
                    comb_outputs_activations
                        .entry(idx)
                        .or_default()
                        .insert(2, None);
                }
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations
                        .entry(dff.d_iv >> 1)
                        .or_default()
                        .insert(dff.en_iv << 1 | (dff.d_iv & 1), None);
                }
            }
        }
        let mut num_duplicate_bits = 0usize;
        for (pin, activations) in &comb_outputs_activations {
            if *pin == 0 {
                num_duplicate_bits += activations.len();
            } else if activations.len() > 1 {
                num_duplicate_bits += activations.len() - 1;
            }
        }
        self.num_duplicate_writeouts = num_duplicate_bits.div_ceil(32) as u32;
        self.comb_outputs_activations = comb_outputs_activations;

        self.num_writeouts = self.num_normal_writeouts
            + self.num_srams
            + self.num_duplicate_writeouts;

        self.after_writeout_pin2pos = self
            .parts_after_writeouts
            .iter()
            .enumerate()
            .filter_map(|(i, &pin)| {
                if pin == usize::MAX {
                    None
                } else {
                    Some((pin, i as u16))
                }
            })
            .collect::<IndexMap<_, _>>();
    }

    /// Looks up how to fetch a pin value given its index-with-invert encoding.
    ///
    /// Returns (permutation_id, invert_bit, set_zero_bit):
    /// - `permutation_id`: The local position in the writeout buffer to read
    ///   from.
    /// - `invert_bit`: Whether to XOR the value (1 if pin_iv has the invert
    ///   flag set).
    /// - `set_zero_bit`: Whether to force the value to zero (1 for constants).
    ///
    /// Special case: pin_iv <= 1 represents constants (0 or 1), handled by
    /// returning permutation_id=0 with set_zero=1 and invert=pin_iv.
    fn query_permute_with_pin_iv(&self, pin_iv: usize) -> (u16, u8, u8) {
        if pin_iv <= 1 {
            // here lsb is just the value so we dont need the & 1
            // constants have 0 permutation id?
            // setzero bit is 1?
            return (0, pin_iv as u8, 1);
        }
        let pos = self.after_writeout_pin2pos.get(&(pin_iv >> 1)).unwrap();
        (*pos, (pin_iv & 1) as u8, 0)
    }

    /// Places a single bit entry in the SRAM/duplicate permutation table.
    ///
    /// The SRAM/duplicate permutation is used by the GPU kernel to gather:
    /// 1. SRAM control inputs (read addr, write addr, write enable, write data)
    /// 2. Duplicated output bits when the same combinational output feeds
    ///    multiple endpoints with different clock enables or polarities.
    ///
    /// Called during `make_inputs_outputs` to populate
    /// `sram_duplicate_permute/inv/set0`.
    fn place_sram_duplicate(
        &mut self,
        pos: usize,
        (perm, inv, set0): (u16, u8, u8),
    ) {
        self.sram_duplicate_permute[pos] = perm;
        set_bit_in_u32(
            &mut self.sram_duplicate_inv[pos >> 5],
            (pos & 31) as u32,
            inv,
        );
        set_bit_in_u32(
            &mut self.sram_duplicate_set0[pos >> 5],
            (pos & 31) as u32,
            set0,
        );
    }

    /// Places a single bit entry in the clock-enable and data-invert tables.
    ///
    /// Each writeout bit has an associated clock enable signal that controls
    /// whether the value is actually written (for DFFs with enable pins).
    /// The data_inv bit tracks whether the output data itself needs
    /// inversion.
    ///
    /// Called during `make_inputs_outputs` to populate `clken_permute/inv/set0`
    /// and `data_inv`.
    fn place_clken_datainv(
        &mut self,
        pos: usize,
        clken_iv_perm: u16,
        clken_iv_inv: u8,
        clken_iv_set0: u8,
        data_inv: u8,
    ) {
        self.clken_permute[pos] = clken_iv_perm;
        set_bit_in_u32(
            &mut self.clken_inv[pos >> 5],
            (pos & 31) as u32,
            clken_iv_inv,
        );
        set_bit_in_u32(
            &mut self.clken_set0[pos >> 5],
            (pos & 31) as u32,
            clken_iv_set0,
        );
        set_bit_in_u32(
            &mut self.data_inv[pos >> 5],
            (pos & 31) as u32,
            data_inv,
        );
    }

    /// Returns the local writeout position for an output bit with a specific
    /// activation.
    ///
    /// An "activation" is the combination of (pin_iv, clken_iv) - the data pin
    /// (with invert) and its clock enable (with invert). Multiple endpoints
    /// may use the same combinational output but with different clock
    /// enables (e.g., two DFFs driven by the same signal but with different
    /// enable conditions). Each unique activation needs its own writeout slot.
    ///
    /// If this activation was already placed, returns the cached position.
    /// Otherwise, allocates a position:
    /// - If this is the "primary" activation (first one for this pin) and the
    ///   pin is not constant, uses the original writeout position from the
    ///   boomerang stage.
    /// - Otherwise, allocates a slot from the duplicate writeout region and
    ///   sets up the sram_duplicate permutation to copy the value there.
    ///
    /// Also populates the clock enable and data inversion tables for the
    /// allocated position.
    fn get_or_place_output_with_activation(
        &mut self,
        pin_iv: usize,
        clken_iv: usize,
    ) -> u16 {
        let (activ_idx, _, pos) = self
            .comb_outputs_activations
            .get(&(pin_iv >> 1))
            .unwrap()
            .get_full(&(clken_iv << 1 | (pin_iv & 1)))
            .unwrap();
        if let Some(pos) = *pos {
            return pos;
        }
        let (clken_iv_perm, clken_iv_inv, clken_iv_set0) =
            self.query_permute_with_pin_iv(clken_iv);
        let is_constant_pin = pin_iv <= 1;
        let data_inv = if is_constant_pin {
            0
        } else {
            (pin_iv & 1) as u8
        };
        let r_pos = if activ_idx == 0 && !is_constant_pin {
            let origpos = *self.after_writeout_pin2pos.get(&(pin_iv >> 1)).unwrap_or_else(
                || panic!("position of pin_iv {} (clken_iv {}) not found.. buggy boomerang, check if netlist and gemparts mismatch.", pin_iv, clken_iv)
            ) as usize;
            self.place_clken_datainv(
                origpos,
                clken_iv_perm,
                clken_iv_inv,
                clken_iv_set0,
                data_inv,
            );
            origpos as u16
        } else {
            self.cnt_placed_duplicate_permute += 1;
            let dup_pos = ((self.num_writeouts - self.num_srams) * 32
                - self.cnt_placed_duplicate_permute)
                as usize;
            let dup_perm_pos =
                ((self.num_srams * 4 + self.num_duplicate_writeouts) * 32
                    - self.cnt_placed_duplicate_permute)
                    as usize;
            assert!(
                dup_perm_pos < 8192,
                "sram duplicate bit larger than expected.."
            );
            let (perm, inv, set0) = if is_constant_pin {
                self.query_permute_with_pin_iv(pin_iv)
            } else {
                let origpos = *self.after_writeout_pin2pos.get(&(pin_iv >> 1)).unwrap_or_else(
                    || panic!("position of pin_iv {} (clken_iv {}) not found.. buggy boomerang, check if netlist and gemparts mismatch.", pin_iv, clken_iv)
                );
                (origpos, 0, 0)
            };
            self.place_sram_duplicate(dup_perm_pos, (perm, inv, set0));
            self.place_clken_datainv(
                dup_pos,
                clken_iv_perm,
                clken_iv_inv,
                clken_iv_set0,
                data_inv,
            );
            dup_pos as u16
        };
        *self
            .comb_outputs_activations
            .get_mut(&(pin_iv >> 1))
            .unwrap()
            .get_mut(&(clken_iv << 1 | (pin_iv & 1)))
            .unwrap() = Some(r_pos);
        r_pos
    }

    /// Populates input/output mappings and builds permutation tables for this
    /// partition.
    ///
    /// Called second during flattening (after `init_afters_writeouts` and state
    /// position allocation) to set up how data flows in and out of this
    /// partition.
    ///
    /// This function processes each endpoint in the partition:
    /// - **RAMBlock**: Sets up SRAM read data as inputs, configures
    ///   permutations for SRAM control signals (read address, write address,
    ///   write enable, write data).
    /// - **PrimaryOutput**: Maps output pins to their writeout positions with
    ///   clock enable = always.
    /// - **StagedIOPin**: Maps intermediate values between major stages.
    /// - **DFF**: Maps D inputs to writeout positions with their clock enable,
    ///   Q outputs to inputs.
    ///
    /// The function builds three permutation tables consumed by the GPU kernel:
    /// - `sram_duplicate_permute/inv/set0`: Permutes SRAM inputs and duplicated
    ///   output pins.
    /// - `clken_permute/inv/set0`: Permutes clock enable signals for
    ///   conditional writes.
    /// - `data_inv`: Tracks which data bits need inversion.
    pub(crate) fn make_inputs_outputs(
        &mut self,
        aig: &AIG,
        staged: &StagedAIG,
        part: &Partition,
        input_map: &mut IndexMap<usize, u32>,
        staged_io_map: &mut IndexMap<usize, u32>,
        output_map: &mut IndexMap<usize, u32>,
    ) {
        self.sram_duplicate_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.sram_duplicate_inv = vec![0u32; NUM_THREADS_V1];
        self.sram_duplicate_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.clken_permute = vec![0; 1 << BOOMERANG_NUM_STAGES];
        self.clken_inv = vec![0u32; NUM_THREADS_V1];
        self.clken_set0 = vec![u32::MAX; NUM_THREADS_V1];
        self.data_inv = vec![0u32; NUM_THREADS_V1];
        self.cnt_placed_duplicate_permute = 0;

        let mut cur_sram_id = 0;
        for &endpt_i in &part.endpoints {
            match staged.get_endpoint_group(aig, endpt_i) {
                EndpointGroup::RAMBlock(ram) => {
                    let sram_rd_data_local_offset = self.num_writeouts as usize
                        - self.num_srams as usize
                        + cur_sram_id as usize;
                    let sram_rd_data_global_start =
                        self.state_start + self.num_writeouts - self.num_srams
                            + cur_sram_id;
                    let (perm_r_en_iv, perm_r_en_iv_inv, perm_r_en_iv_set0) =
                        self.query_permute_with_pin_iv(ram.port_r_en_iv);
                    for k in 0..32 {
                        let d = ram.port_r_rd_data[k];
                        if d == usize::MAX {
                            continue;
                        }
                        input_map.insert(
                            d,
                            sram_rd_data_global_start * 32 + k as u32,
                        );
                        output_map.insert(
                            d << 1,
                            sram_rd_data_global_start * 32 + k as u32,
                        );
                        self.place_clken_datainv(
                            sram_rd_data_local_offset * 32 + k,
                            perm_r_en_iv,
                            perm_r_en_iv_inv,
                            perm_r_en_iv_set0,
                            0,
                        );
                    }
                    let sram_input_perm_st = (cur_sram_id * 32 * 4) as usize;
                    for k in 0..13 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + k,
                            self.query_permute_with_pin_iv(
                                ram.port_r_addr_iv[k],
                            ),
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 16 + k,
                            self.query_permute_with_pin_iv(
                                ram.port_w_addr_iv[k],
                            ),
                        );
                    }
                    for k in 0..32 {
                        self.place_sram_duplicate(
                            sram_input_perm_st + 32 + k,
                            self.query_permute_with_pin_iv(
                                ram.port_w_wr_en_iv[k],
                            ),
                        );
                        self.place_sram_duplicate(
                            sram_input_perm_st + 64 + k,
                            self.query_permute_with_pin_iv(
                                ram.port_w_wr_data_iv[k],
                            ),
                        );
                    }
                    cur_sram_id += 1;
                }
                EndpointGroup::PrimaryOutput(idx_iv) => {
                    assert!(idx_iv != 0, "primary output can't be 0");
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(idx_iv, 1)
                            as u32;
                    output_map.insert(idx_iv, pos);
                }
                EndpointGroup::StagedIOPin(idx) => {
                    assert!(idx != 0, "staged IO pin can't be 0");
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(idx << 1, 1)
                            as u32;
                    staged_io_map.insert(idx, pos);
                }
                EndpointGroup::DFF(dff) => {
                    if dff.d_iv <= 1 {
                        tracing::warn!(
                            d_iv = dff.d_iv,
                            "dff d_iv is constant, continuing with constant data"
                        );
                    }
                    let pos = self.state_start * 32
                        + self.get_or_place_output_with_activation(
                            dff.d_iv, dff.en_iv,
                        ) as u32;
                    output_map.insert(dff.d_iv, pos);
                    input_map.insert(dff.q, pos);
                }
            }
        }
        assert_eq!(cur_sram_id, self.num_srams);
        assert_eq!(
            self.cnt_placed_duplicate_permute.div_ceil(32),
            self.num_duplicate_writeouts
        );

        // println!("test clken_permute: {:?}, wos (w/o sram or dup): {:?}",
        // self.clken_permute, self.parts_after_writeouts);
    }

    /// Generates the GPU kernel script (instruction stream) for this partition.
    ///
    /// Called third during flattening (after `make_inputs_outputs`) to produce
    /// the final binary data that the GPU partition executor consumes.
    ///
    /// The script format (documented in `FlattenedScriptV1::blocks_data`):
    /// 1. **Metadata** (256 words): Boomerang stage count, writeout
    ///    count/offset, SRAM info, global read rounds, duplicate writeout
    ///    count, writeout locations.
    /// 2. **Global read permutation**: For each of `num_global_stages` rounds,
    ///    specifies which 32-bit words to read from global state and which bits
    ///    within them are valid.
    /// 3. **Boomerang stages** (repeated): For each stage:
    ///    - Local shuffle permutation (which local positions feed into each AND
    ///      gate)
    ///    - XOR-A, XOR-B, OR-B masks (encode AND gate inversions and
    ///      passthroughs)
    /// 4. **SRAM/duplicate permutation**: How to gather SRAM inputs and
    ///    duplicated outputs.
    /// 5. **Clock enable permutation**: How to gather clock enable signals and
    ///    data inversions.
    ///
    /// The `input_map` and `staged_io_map` are used to resolve where each
    /// partition input lives in global state memory.
    pub(crate) fn build_script(
        &self,
        aig: &AIG,
        part: &Partition,
        input_map: &IndexMap<usize, u32>,
        staged_io_map: &IndexMap<usize, u32>,
    ) -> Vec<u32> {
        // metadata
        let mut script = vec![
            part.stages.len() as u32,
            0,
            self.num_writeouts,
            self.state_start,
            self.num_srams,
            self.sram_start,
            0, // [6]=num global read rounds, assigned later
            self.num_duplicate_writeouts,
        ];
        // padding
        while script.len() < 128 {
            script.push(0);
        }
        // final 128: write-out locations
        // compressed 2-1
        let mut last_wo = u32::MAX;
        for (j, bs) in part.stages.iter().enumerate() {
            for &wo in &bs.write_outs {
                let cur_wo = (j as u32) << 8 | (wo as u32);
                if last_wo == u32::MAX {
                    last_wo = cur_wo;
                } else {
                    script.push(last_wo | (cur_wo << 16));
                    last_wo = u32::MAX;
                }
            }
        }
        if last_wo != u32::MAX {
            script.push(last_wo | (((1 << 16) - 1) << 16));
        }
        while script.len() < 256 {
            script.push(u32::MAX);
        }
        // read global (256x32)
        let mut inputs_taken = BTreeMap::<u32, u32>::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX {
                continue;
            }
            match input_map.get(&inp) {
                Some(&pos) => {
                    *inputs_taken.entry(pos >> 5).or_default() |=
                        1 << (pos & 31);
                }
                None => match staged_io_map.get(&inp) {
                    Some(&pos) => {
                        *inputs_taken
                            .entry((pos >> 5) | (1u32 << 31))
                            .or_default() |= 1 << (pos & 31);
                    }
                    None => {
                        panic!(
                            "cannot find input pin {}, driver: {:?}, in either primary inputs or staged IOs",
                            inp, aig.drivers[inp]
                        );
                    }
                },
            }
        }
        let rounds_idx_masks = map_global_read_to_rounds(&inputs_taken);
        let num_global_stages =
            rounds_idx_masks.iter().map(|v| v.len()).max().unwrap() as u32;
        script[6] = num_global_stages;
        assert_eq!(script.len(), NUM_THREADS_V1);
        let global_perm_start = script.len();
        script.extend(
            (0..(2 * num_global_stages as usize * NUM_THREADS_V1)).map(|_| 0),
        );
        for (i, v) in rounds_idx_masks.iter().enumerate() {
            for (round, &(idx, mask)) in v.iter().enumerate() {
                script[global_perm_start
                    + NUM_THREADS_V1 * 2 * round
                    + (i * 2)] = idx;
                script[global_perm_start
                    + NUM_THREADS_V1 * 2 * round
                    + (i * 2 + 1)] = mask;
            }
        }

        let outputpos2localpos = rounds_idx_masks
            .iter()
            .enumerate()
            .flat_map(|(local_i, v)| {
                let mut local_op2lp = Vec::with_capacity(32);
                let mut bit_id = 0;
                for &(idx, mask) in v.iter().rev() {
                    let is_staged_io = (idx >> 31) != 0;
                    for k in (0..32).rev() {
                        if (mask >> k & 1) != 0 {
                            local_op2lp.push((
                                (is_staged_io, idx << 5 | k),
                                (local_i * 32 + bit_id) as u16,
                            ));
                            bit_id += 1;
                        }
                    }
                }
                assert!(bit_id <= 32);
                local_op2lp.into_iter()
            })
            .collect::<IndexMap<_, _>>();

        let mut last_pin2localpos = IndexMap::new();
        for &inp in &part.stages[0].hier[0] {
            if inp == usize::MAX {
                continue;
            }
            let pos = match input_map.get(&inp) {
                Some(&pos) => (false, pos),
                None => (true, *staged_io_map.get(&inp).unwrap()),
            };
            last_pin2localpos
                .insert(inp, *outputpos2localpos.get(&pos).unwrap());
        }

        // boomerang sections start
        for (bs_i, bs) in part.stages.iter().enumerate() {
            let bs_perm = bs.hier[0]
                .iter()
                .map(|&pin| {
                    if pin == usize::MAX {
                        0
                    } else {
                        *last_pin2localpos.get(&pin).unwrap()
                    }
                })
                .collect::<Vec<_>>();

            let mut bs_xora = vec![0u32; NUM_THREADS_V1];
            let mut bs_xorb = vec![0u32; NUM_THREADS_V1];
            let mut bs_orb = vec![0u32; NUM_THREADS_V1];
            for hi in 1..bs.hier.len() {
                let hi_len = bs.hier[hi].len();
                for j in 0..hi_len {
                    let out = bs.hier[hi][j];
                    let a = bs.hier[hi - 1][j];
                    let b = bs.hier[hi - 1][j + hi_len];
                    if out == usize::MAX {
                        continue;
                    }
                    if out == a {
                        bs_orb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                        continue;
                    }
                    let (a_iv, b_iv) = match aig.drivers[out] {
                        DriverType::AndGate(a_iv, b_iv) => (a_iv, b_iv),
                        _ => unreachable!(),
                    };
                    assert_eq!(a_iv >> 1, a);
                    assert_eq!(b_iv >> 1, b);
                    if (a_iv & 1) != 0 {
                        bs_xora[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                    if (b_iv & 1) != 0 {
                        bs_xorb[(hi_len + j) >> 5] |= 1 << ((hi_len + j) & 31);
                    }
                }
            }

            for k in 0..4 {
                for i in ((k * 8)..bs_perm.len()).step_by(32) {
                    script.push(
                        (bs_perm[i] as u32) | (bs_perm[i + 1] as u32) << 16,
                    );
                    script.push(
                        (bs_perm[i + 2] as u32) | (bs_perm[i + 3] as u32) << 16,
                    );
                    script.push(
                        (bs_perm[i + 4] as u32) | (bs_perm[i + 5] as u32) << 16,
                    );
                    script.push(
                        (bs_perm[i + 6] as u32) | (bs_perm[i + 7] as u32) << 16,
                    );
                }
            }
            for i in 0..NUM_THREADS_V1 {
                script.push(bs_xora[i]);
                script.push(bs_xorb[i]);
                script.push(bs_orb[i]);
                script.push(0);
            }

            last_pin2localpos = self.afters[bs_i]
                .iter()
                .enumerate()
                .filter_map(|(i, &pin)| {
                    if pin == usize::MAX {
                        None
                    } else {
                        Some((pin, i as u16))
                    }
                })
                .collect::<IndexMap<_, _>>();
        }

        // sram worker
        for k in 0..4 {
            for i in ((k * 8)..self.sram_duplicate_permute.len()).step_by(32) {
                script.push(
                    (self.sram_duplicate_permute[i] as u32)
                        | (self.sram_duplicate_permute[i + 1] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 2] as u32)
                        | (self.sram_duplicate_permute[i + 3] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 4] as u32)
                        | (self.sram_duplicate_permute[i + 5] as u32) << 16,
                );
                script.push(
                    (self.sram_duplicate_permute[i + 6] as u32)
                        | (self.sram_duplicate_permute[i + 7] as u32) << 16,
                );
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.sram_duplicate_inv[i]);
            script.push(self.sram_duplicate_set0[i]);
            script.push(0);
            script.push(0);
        }
        // clock enable signal
        for k in 0..4 {
            for i in ((k * 8)..self.clken_permute.len()).step_by(32) {
                script.push(
                    (self.clken_permute[i] as u32)
                        | (self.clken_permute[i + 1] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 2] as u32)
                        | (self.clken_permute[i + 3] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 4] as u32)
                        | (self.clken_permute[i + 5] as u32) << 16,
                );
                script.push(
                    (self.clken_permute[i + 6] as u32)
                        | (self.clken_permute[i + 7] as u32) << 16,
                );
            }
        }
        for i in 0..NUM_THREADS_V1 {
            script.push(self.clken_inv[i]);
            script.push(self.clken_set0[i]);
            script.push(self.data_inv[i]);
            script.push(0);
        }

        script
    }
}
