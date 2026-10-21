// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! Partition scheduler and flattener

use std::collections::BTreeMap;

use aig::{AIG, aigpdk::AIGPDK_SRAM_ADDR_WIDTH};
use indexmap::IndexMap;
use staging::StagedAIG;

use crate::{
    FlattenedScriptV1,
    parts::FlatteningPart,
    pe::{BOOMERANG_NUM_STAGES, Partition},
};

pub(crate) const NUM_THREADS_V1: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

pub(crate) fn map_global_read_to_rounds(
    inputs_taken: &BTreeMap<u32, u32>,
) -> Vec<Vec<(u32, u32)>> {
    let inputs_taken = inputs_taken
        .iter()
        .map(|(&a, &b)| (a, b))
        .collect::<Vec<_>>();
    // the larger the sorting chunk size, the better the successful chance,
    // but the less efficient due to worse cache coherency.
    let mut chunk_size = inputs_taken.len();
    while chunk_size >= 1 {
        let mut slices = inputs_taken.chunks(chunk_size).collect::<Vec<_>>();
        slices.sort_by_cached_key(|&slice| {
            u32::MAX
                - slice.iter().map(|(_, mask)| mask.count_ones()).sum::<u32>()
        });
        let mut rounds_idx_masks: Vec<Vec<(u32, u32)>> =
            vec![vec![]; NUM_THREADS_V1];
        let mut round_map_j = 0;
        let mut fail = false;
        for slice in slices {
            for &(offset, mask) in slice {
                let wrap_fail_j = round_map_j;
                while rounds_idx_masks[round_map_j]
                    .iter()
                    .map(|(_, mask)| mask.count_ones())
                    .sum::<u32>()
                    + mask.count_ones()
                    > 32
                {
                    round_map_j += 1;
                    if round_map_j == NUM_THREADS_V1 {
                        round_map_j = 0;
                    }
                    if round_map_j == wrap_fail_j {
                        // panic!("failed to map at part {} mem offset {}", i,
                        // offset);
                        fail = true;
                        break;
                    }
                }
                if fail {
                    break;
                }
                rounds_idx_masks[round_map_j].push((offset, mask));
                round_map_j += 1;
                if round_map_j == NUM_THREADS_V1 {
                    round_map_j = 0;
                }
            }
            if fail {
                break;
            }
        }
        if !fail {
            // let max_rounds = rounds_idx_masks.iter().map(|v|
            // v.len()).max().unwrap();
            // println!("max_rounds: {}, round_map_j: {}, inputs_taken len {}",
            // max_rounds, round_map_j, inputs_taken.len());
            return rounds_idx_masks;
        }
        chunk_size /= 2;
    }
    panic!("cannot map global init to any multiples of rounds.");
}

pub(crate) fn build_flattened_script_v1(
    aig: &AIG,
    stageds: &[&StagedAIG],
    parts_in_stages: &[&[Partition]],
    num_blocks: usize,
    input_layout: Vec<usize>,
) -> FlattenedScriptV1 {
    // determine the output position.
    // this is the prerequisite for generating the read
    // permutations and more.
    // input map:
    // locate input pins and FF/SRAM Q's - for partition input
    // output map:
    // locate primary outputs - for circuit outs
    // staged io map:
    // store intermediate nodes between major stages
    let mut input_map = IndexMap::new();
    let mut output_map = IndexMap::new();
    let mut staged_io_map = IndexMap::new();
    for (i, &input) in input_layout.iter().enumerate() {
        if input == usize::MAX {
            continue;
        }
        input_map.insert(input, i as u32);
    }

    let num_major_stages = parts_in_stages.len();

    let states_start = input_layout.len().div_ceil(32) as u32;
    let mut sum_state_start = states_start;
    let mut sum_srams_start = 0;

    // enumerate all major stages and build them one by one.

    // #[derive(Debug, Clone, Default)]
    // struct FlatteningStage {
    //     blocks_parts: Vec<Vec<usize>>,
    //     flattening_parts: Vec<FlatteningPart>,
    //     parts_data_split: Vec<Vec<u32>>,
    // }
    // let mut flattening_stages =
    //     Vec::<FlatteningStage>::with_capacity(num_major_stages);

    // assemble script per block.
    let mut blocks_data = Vec::new();
    let mut blocks_start =
        Vec::<usize>::with_capacity(num_blocks * num_major_stages + 1);
    let mut stages_blocks_parts = Vec::new();
    let mut stages_flattening_parts = Vec::new();

    for (i, (init_parts, &staged)) in parts_in_stages
        .iter()
        .copied()
        .zip(stageds.iter())
        .enumerate()
    {
        // first arrange parts onto blocks.
        let mut blocks_parts = vec![vec![]; num_blocks];
        let mut tot_nstages_blocks = vec![0; num_blocks];
        // below models the fixed pre&post-cost for each executor
        let executor_fixed_cost = 3;
        // masonry layout of blocks. assume parts are sorted with
        // decreasing order of #stages.
        for i in 0..init_parts.len().min(num_blocks) {
            blocks_parts[i].push(i);
            tot_nstages_blocks[i] =
                init_parts[i].stages.len() + executor_fixed_cost;
        }
        for (i, item) in init_parts.iter().enumerate().skip(num_blocks) {
            let put = tot_nstages_blocks
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.cmp(b))
                .unwrap()
                .0;
            blocks_parts[put].push(i);
            tot_nstages_blocks[put] += item.stages.len() + executor_fixed_cost;
        }
        // tracing::debug!("blocks_parts: {:?}", blocks_parts);
        tracing::debug!(
            "major stage {}: max total boomerang depth (w/ cost) {}",
            i,
            tot_nstages_blocks.iter().copied().max().unwrap()
        );

        // the intermediates for parts being flattened
        let mut flattening_parts: Vec<FlatteningPart> =
            vec![Default::default(); init_parts.len()];

        // basic index preprocessing for stages
        for i in 0..init_parts.len() {
            flattening_parts[i].init_afters_writeouts(
                aig,
                staged,
                &init_parts[i],
            );
        }

        // allocate output state positions for all srams,
        // in the order of block affinity.
        for block in &blocks_parts {
            for &part_id in block {
                flattening_parts[part_id].state_start = sum_state_start;
                sum_state_start += flattening_parts[part_id].num_writeouts;
                flattening_parts[part_id].sram_start = sum_srams_start;
                sum_srams_start += flattening_parts[part_id].num_srams
                    * (1 << AIGPDK_SRAM_ADDR_WIDTH);
            }
        }

        // besides input ports, we also have outputs from partitions.
        // they include original-placed comb output pins,
        // copied pins for different FF activation,
        // and SRAM read outputs.
        for part_id in 0..init_parts.len() {
            // tracing::debug!("initializing output for part {}", part_id);
            flattening_parts[part_id].make_inputs_outputs(
                aig,
                staged,
                &init_parts[part_id],
                &mut input_map,
                &mut staged_io_map,
                &mut output_map,
            );
        }
        stages_blocks_parts.push(blocks_parts);
        stages_flattening_parts.push(flattening_parts);
    }

    for ((blocks_parts, flattening_parts), init_parts) in stages_blocks_parts
        .iter()
        .zip(stages_flattening_parts.iter_mut())
        .zip(parts_in_stages.iter().copied())
    {
        // build script per part. we will later assemble them to blocks.
        let mut parts_data_split = vec![vec![]; init_parts.len()];
        for part_id in 0..init_parts.len() {
            // tracing::debug!("building script for part {}", part_id);
            parts_data_split[part_id] = flattening_parts[part_id].build_script(
                aig,
                &init_parts[part_id],
                &input_map,
                &staged_io_map,
            );
        }

        for block in blocks_parts.iter().take(num_blocks) {
            blocks_start.push(blocks_data.len());
            if block.is_empty() {
                let mut dummy = vec![0; NUM_THREADS_V1];
                dummy[1] = 1;
                blocks_data.extend(dummy.into_iter());
            } else {
                let num_parts = block.len();
                let mut last_part_st = usize::MAX;
                for (i, &part_id) in block.iter().enumerate() {
                    if i == num_parts - 1 {
                        last_part_st = blocks_data.len();
                    }
                    blocks_data
                        .extend(parts_data_split[part_id].iter().copied());
                }
                assert_ne!(last_part_st, usize::MAX);
                blocks_data[last_part_st + 1] = 1;
            }
        }
    }
    blocks_start.push(blocks_data.len());
    blocks_data.extend((0..NUM_THREADS_V1 * 8).map(|_| 0)); // padding

    tracing::info!(
        "Built script for {} blocks, reg/io state size {}, sram size {}, script size {}",
        num_blocks,
        sum_state_start,
        sum_srams_start,
        blocks_data.len()
    );

    FlattenedScriptV1 {
        num_blocks,
        num_major_stages,
        blocks_start: blocks_start.into(),
        blocks_data: blocks_data.into(),
        reg_io_state_size: sum_state_start,
        sram_storage_size: sum_srams_start,
        input_layout,
        input_map,
        output_map,
        stages_blocks_parts,
    }
}
