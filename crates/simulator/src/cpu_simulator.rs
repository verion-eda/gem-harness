// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
use aig::aigpdk::AIGPDK_SRAM_SIZE;
use gem_cuda::ucci::ClockEdgeInfo;

/// CPU simulation wrapper that executes all blocks sequentially.
pub(crate) struct CpuSimulator {}

impl CpuSimulator {
    pub(crate) fn simulate_single_cycle(
        num_blocks: usize,
        num_major_stages: usize,
        blocks_start: &[usize],
        blocks_data: &[u32],
        sram_data: &mut [u32],
        _state_size: usize,
        input_state: &mut [u32],
        output_state: &mut [u32],
        clock_edges: &[ClockEdgeInfo],
    ) {
        // Detect clock edges by comparing input_state (new) vs output_state
        // (old)
        for edge in clock_edges {
            // Clear old edge flags
            if edge.posedge_word != u32::MAX {
                input_state[edge.posedge_word as usize] &= !edge.posedge_mask;
            }
            if edge.negedge_word != u32::MAX {
                input_state[edge.negedge_word as usize] &= !edge.negedge_mask;
            }

            let new_val =
                input_state[edge.clock_word as usize] & edge.clock_mask;
            let old_val =
                output_state[edge.clock_word as usize] & edge.clock_mask;

            if old_val == 0 && new_val != 0 {
                // Rising edge
                if edge.posedge_word != u32::MAX {
                    input_state[edge.posedge_word as usize] |=
                        edge.posedge_mask;
                }
            } else if old_val != 0 && new_val == 0 {
                // Falling edge
                if edge.negedge_word != u32::MAX {
                    input_state[edge.negedge_word as usize] |=
                        edge.negedge_mask;
                }
            }
        }

        output_state.copy_from_slice(input_state);

        for stage_i in 0..num_major_stages {
            for blk_i in 0..num_blocks {
                let start = blocks_start[stage_i * num_blocks + blk_i];
                let end = blocks_start[stage_i * num_blocks + blk_i + 1];
                simulate_block_v1(
                    &blocks_data[start..end],
                    input_state,
                    output_state,
                    sram_data,
                    false,
                );
            }
        }
    }
}

/// CPU prototype partition executor for script version 1.
fn simulate_block_v1(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
    debug_verbose: bool,
) {
    let mut script_pi = 0;
    loop {
        let num_stages = script[script_pi];
        let is_last_part = script[script_pi + 1];
        let num_ios = script[script_pi + 2];
        let io_offset = script[script_pi + 3];
        let num_srams = script[script_pi + 4];
        let sram_offset = script[script_pi + 5];
        let num_global_read_rounds = script[script_pi + 6];
        let num_output_duplicates = script[script_pi + 7];
        let mut writeout_hooks = vec![0; 256];
        for i in 0..128 {
            let t = script[script_pi + 128 + i];
            writeout_hooks[i * 2] = (t & ((1 << 16) - 1)) as u16;
            writeout_hooks[i * 2 + 1] = (t >> 16) as u16;
        }
        if num_stages == 0 {
            script_pi += 256;
            break;
        }
        script_pi += 256;
        let mut writeouts = vec![0u32; num_ios as usize];

        let mut state = vec![0u32; 256];
        for _gr_i in 0..num_global_read_rounds {
            for i in 0..256 {
                let mut cur_state = state[i];
                let idx = script[script_pi + (i * 2)];
                let mut mask = script[script_pi + (i * 2 + 1)];
                if mask == 0 {
                    continue;
                }
                let value = match (idx >> 31) != 0 {
                    false => input_state[idx as usize],
                    true => output_state[(idx ^ (1 << 31)) as usize],
                };
                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 {
                        cur_state |= 1;
                    }
                    mask ^= lowbit;
                }
                state[i] = cur_state;
            }
            script_pi += 256 * 2;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 0");
            println!("global read states:");
            for (i, item) in state.iter().enumerate().take(256) {
                println!(" [{}] = {}", i, item);
            }
        }

        for bs_i in 0..num_stages {
            let mut hier_inputs = vec![0; 256];
            let mut hier_flag_xora = vec![0; 256];
            let mut hier_flag_xorb = vec![0; 256];
            let mut hier_flag_orb = vec![0; 256];
            for k_outer in 0..4 {
                for i in 0..256 {
                    for k_inner in 0..4 {
                        let k: usize = k_outer * 4 + k_inner;
                        let t_shuffle = script[script_pi + i * 4 + k_inner];
                        let t_shuffle_1_idx =
                            (t_shuffle & ((1 << 16) - 1)) as u16;
                        let t_shuffle_2_idx = (t_shuffle >> 16) as u16;
                        hier_inputs[i] |= (state
                            [(t_shuffle_1_idx >> 5) as usize]
                            >> (t_shuffle_1_idx & 31)
                            & 1)
                            << (k * 2);
                        hier_inputs[i] |= (state
                            [(t_shuffle_2_idx >> 5) as usize]
                            >> (t_shuffle_2_idx & 31)
                            & 1)
                            << (k * 2 + 1);
                    }
                }
                script_pi += 256 * 4;
            }
            for i in 0..256 {
                hier_flag_xora[i] = script[script_pi + i * 4];
                hier_flag_xorb[i] = script[script_pi + i * 4 + 1];
                hier_flag_orb[i] = script[script_pi + i * 4 + 2];
            }
            script_pi += 256 * 4;

            if debug_verbose {
                println!("debug_verbose STAGE 1.1 bs_i {bs_i}");
                println!("after local shuffle:");
                for (i, item) in hier_inputs.iter().enumerate().take(256) {
                    println!(" [{}] = {}", i, item);
                }
            }

            for i in 0..128 {
                let a = hier_inputs[i];
                let b = hier_inputs[128 + i];
                let xora = hier_flag_xora[128 + i];
                let xorb = hier_flag_xorb[128 + i];
                let orb = hier_flag_orb[128 + i];
                let ret = (a ^ xora) & ((b ^ xorb) | orb);
                hier_inputs[128 + i] = ret;
            }
            for hi in 1..=7 {
                let hier_width = 1 << (7 - hi);
                for i in 0..hier_width {
                    let a = hier_inputs[hier_width * 2 + i];
                    let b = hier_inputs[hier_width * 3 + i];
                    let xora = hier_flag_xora[hier_width + i];
                    let xorb = hier_flag_xorb[hier_width + i];
                    let orb = hier_flag_orb[hier_width + i];
                    let ret = (a ^ xora) & ((b ^ xorb) | orb);
                    hier_inputs[hier_width + i] = ret;
                }
            }
            let v1 = hier_inputs[1];
            let xora = hier_flag_xora[0];
            let xorb = hier_flag_xorb[0];
            let orb = hier_flag_orb[0];
            let r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;
            let r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;
            let r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;
            let r11 =
                ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;
            let r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;
            hier_inputs[0] = r8 | r9 | r10 | r11 | r12;

            state = hier_inputs;

            if debug_verbose {
                println!("debug_verbose STAGE 1.2 bs_i {bs_i}");
                println!("after and-invert:");
                for (i, item) in state.iter().enumerate().take(256) {
                    println!(" [{}] = {}", i, item);
                }
            }

            for i in 0..256 {
                let hooki = writeout_hooks[i];
                if (hooki >> 8) as u32 == bs_i {
                    writeouts[i] = state[(hooki & 255) as usize];
                }
            }
        }

        let mut sram_duplicate_perm =
            vec![0u32; (num_srams * 4 + num_output_duplicates) as usize];
        for k_outer in 0..4 {
            for i in 0..(num_srams * 4 + num_output_duplicates) {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle =
                        script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);
                    let t_shuffle_2_idx = t_shuffle >> 16;
                    sram_duplicate_perm[i as usize] |= (writeouts
                        [(t_shuffle_1_idx >> 5) as usize]
                        >> (t_shuffle_1_idx & 31)
                        & 1)
                        << (k * 2);
                    sram_duplicate_perm[i as usize] |= (writeouts
                        [(t_shuffle_2_idx >> 5) as usize]
                        >> (t_shuffle_2_idx & 31)
                        & 1)
                        << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..(num_srams * 4 + num_output_duplicates) as usize {
            sram_duplicate_perm[i] &= !script[script_pi + i * 4 + 1];
            sram_duplicate_perm[i] ^= script[script_pi + i * 4];
        }
        script_pi += 256 * 4;

        for sram_i_u32 in 0..num_srams {
            let sram_i = sram_i_u32 as usize;
            let addrs = sram_duplicate_perm[sram_i * 4];
            let port_r_addr_iv = addrs & 0xffff;
            let port_w_addr_iv = (addrs & 0xffff0000) >> 16;
            let port_w_wr_en = sram_duplicate_perm[sram_i * 4 + 1];
            let port_w_wr_data_iv = sram_duplicate_perm[sram_i * 4 + 2];

            let sram_st = sram_offset as usize + sram_i * AIGPDK_SRAM_SIZE;
            let sram_ed = sram_st + AIGPDK_SRAM_SIZE;
            let ram = &mut sram_data[sram_st..sram_ed];
            let r = ram[port_r_addr_iv as usize];
            let w0 = ram[port_w_addr_iv as usize];
            writeouts[(num_ios - num_srams + sram_i_u32) as usize] = r;
            ram[port_w_addr_iv as usize] =
                (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
        }

        for i in 0..num_output_duplicates {
            writeouts
                [(num_ios - num_srams - num_output_duplicates + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + i) as usize];
        }

        if debug_verbose {
            println!("debug_verbose STAGE 2");
            println!("before writeout_inv:");
            for (i, item) in writeouts.iter().enumerate().take(256) {
                println!(
                    " [{}] = {}",
                    i,
                    if i < num_ios as usize { *item } else { 0 }
                );
            }
        }

        let mut clken_perm = vec![0u32; num_ios as usize];
        let writeouts_for_clken = writeouts.clone();
        for k_outer in 0..4 {
            for i in 0..num_ios {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle =
                        script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);
                    let t_shuffle_2_idx = t_shuffle >> 16;
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_1_idx >> 5) as usize]
                        >> (t_shuffle_1_idx & 31)
                        & 1)
                        << (k * 2);
                    clken_perm[i as usize] |= (writeouts_for_clken
                        [(t_shuffle_2_idx >> 5) as usize]
                        >> (t_shuffle_2_idx & 31)
                        & 1)
                        << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }
        script_pi += 256 * 4;

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 3");
            println!("final writeout:");
            for i in 0..num_ios {
                println!(
                    " [{}] [global {}] = {}",
                    i,
                    io_offset + i,
                    output_state[(io_offset + i) as usize]
                );
            }
        }

        if is_last_part != 0 {
            break;
        }
    }
    assert_eq!(script_pi, script.len());
}
