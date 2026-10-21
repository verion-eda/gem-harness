// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! Lower a partitioned AIG into the VLIW bitstream the GPU executes.
//!
//! Input is the output of [`repcut`]: an [`AIG`] clustered into partitions,
//! each sized to fit on one Boolean processor (256 CUDA threads). This
//! crate emits a [`FlattenedScriptV1`] containing per-PE instruction
//! streams, the Boomerang hierarchy that schedules them, and memory
//! layout for endpoints. The script is consumed by [`gem-cuda`] at
//! runtime; [`compile`] is the CLI that drives this crate.
//!
//! # Key types
//!
//! - [`FlattenedScriptV1`] — the serialized output; what `.gemparts` files on
//!   disk decode to.
//! - [`pe::Partition`] — one assigned partition; produced by
//!   [`pe::process_partitions`].
//!
//! [`repcut`]: ../repcut/
//! [`gem-cuda`]: ../gem_cuda/
//! [`compile`]: ../compile/

use aig::AIG;
use indexmap::IndexMap;
use serde::{Serialize, Serializer};
use staging::StagedAIG;
use ulib::UVec;

use crate::pe::Partition;

mod internal;
mod parts;
pub mod pe;

/// Custom serde serializer for `UVec` fields. Used for regression tests only.
///
/// `UVec` is a unified CPU/GPU vector from `ulib` that doesn't implement
/// `Serialize`. Since `UVec<T>` implements `Deref<Target = [T]>`, we can borrow
/// it as a slice and serialize that. This only works for serialization (not
/// deserialization), which is sufficient for snapshot testing.
fn serialize_uvec<T, S>(
    uvec: &UVec<T>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    T: ulib::UniversalCopy + Serialize,
    S: Serializer,
{
    let slice: &[T] = uvec;
    slice.serialize(serializer)
}

/// A flattened script, for partition executor version 1.
/// See [FlattenedScriptV1::blocks_data] for the format details.
///
/// Generally, a script contains a number of major stages.
/// Each stage consists of the same number of blocks.
/// Each block contains a list of flattened partitions.
#[derive(Serialize)]
pub struct FlattenedScriptV1 {
    /// the number of blocks
    pub num_blocks: usize,
    /// the number of major stages
    pub num_major_stages: usize,
    /// the CSR start indices of stages and blocks.
    ///
    /// this length is num_blocks * num_major_stages + 1
    #[serde(serialize_with = "serialize_uvec")]
    pub blocks_start: UVec<usize>,
    /// the partition instructions.
    ///
    /// the instructions follow a special format.
    /// it consists of zero or more partitions.
    /// 1. [256 bits] metadata section:
    ///    - [32 bit] the number of boomerang stages: if this is zero, the
    ///      stage will not run. only happens when the block has no partition
    ///      mapped onto it.
    ///    - [32 bit] is this the last boomerang stage? only 0 or 1
    ///    - [32 bit] the number of valid write-outs
    ///    - [32 bit] the write-out destination offset (at this offset, we put
    ///      all FFs and other outputs.)
    ///    - [(32 bit, 32 bit)] the srams count and offsets: (count, memory
    ///      offset for the first mem)
    ///    - [32 bit] the number of global read rounds
    ///    - [32 bit] the number of output-duplicate writeouts: this is used
    ///      when one output pin is used by either \<both output and FFs\>, or
    ///      \<multiple FFs with different enabling conditions\>.
    ///    - padding towards 128
    ///    - the location of early write-outs, excl. mem. `256 * [#boomstage &
    ///      0~256 id]`, compressed to 128
    /// 2. initial read-global permutation \[2x\]*rounds 32-bit indices*1 for
    ///    each of the 256 threads. 32-bit valid mask for each of the 256
    ///    threads. if the valid mask is zero, the memory is not read. the
    ///    result should be stored "like pext instruction", but "reversed", and
    ///    then appended to the low bits in each round. the index is encoded
    ///    with a type bit at the highest bit:
    ///      - if it is 0: it means it is offset from previous iteration.
    ///      - if it is 1: it is offset from current iteration which means it
    ///        is an intermediate value coming from the same cycle but a
    ///        previous major stage.
    /// 3. boomerang sections, repeat below N*16
    ///    1. local shuffle permutation 32-bit indices * 16 for each of the 256
    ///       threads.
    ///    2. input (with inv) * 8192bits * (3+1padding) 32-bit * 256 threads *
    ///       (3+1): xora, xorb, orb, \[0 padding\] 0xy: and gate, out =
    ///       (a^x)&(b^y). 100: passthrough, out = a. 111: invalid, can be
    ///       skipped.
    ///    3. write out, according to rest
    /// 4. global write-outs.
    ///    1. sram & additional endpoint copy permutations, inv. \[16x\]. only
    ///       the inputs within sram and endpoint copy range will be
    ///       considered. followed by \[4x\] invert, set0, and two 0 paddings.
    ///    2. permutation for the write-out enabler pins, inv. \[16\] include
    ///       itself inv and data inv. followed by \[3(+1padding)x\]: clock
    ///       invert, clock set0, data invert, and 0 padding
    ///    3. commit the write-out
    #[serde(serialize_with = "serialize_uvec")]
    pub blocks_data: UVec<u32>,
    /// the state size including DFF and I/O states only.
    ///
    /// the inputs are always at front.
    pub reg_io_state_size: u32,
    /// the u32 array length for storing SRAMs.
    pub sram_storage_size: u32,
    /// expected input AIG pins layout
    pub input_layout: Vec<usize>,
    /// maps from primary outputs, FF:D and SRAM:PORT_R_RD_DATA AIG pins
    /// to state offset, index with invert.
    pub output_map: IndexMap<usize, u32>,
    /// maps from primary inputs, FF:Q/SRAM:* input AIG pins to state offset,
    /// index WITHOUT invert.
    pub input_map: IndexMap<usize, u32>,
    /// (for debug purpose) the relation between major stage, block and
    /// part indices as given in construction.
    pub stages_blocks_parts: Vec<Vec<Vec<usize>>>,
}

impl FlattenedScriptV1 {
    /// build a flattened script.
    ///
    /// `init_parts` give the partitions to flatten.
    /// it is better sorted in advance in descending order
    /// of #layers for better duty cycling.
    ///
    /// `num_blocks` should be set to the hardware allowances,
    /// i.e. the number of SMs in your GPU.
    /// for example, A100 should set it to 108.
    ///
    /// `input_layout` should give the expected primary input
    /// memory layout, each one is an AIG bit index.
    /// padding bits should be set to usize::MAX.
    pub fn from(
        aig: &AIG,
        stageds: &[&StagedAIG],
        parts_in_stages: &[&[Partition]],
        num_blocks: usize,
        input_layout: Vec<usize>,
    ) -> FlattenedScriptV1 {
        internal::build_flattened_script_v1(
            aig,
            stageds,
            parts_in_stages,
            num_blocks,
            input_layout,
        )
    }
}
