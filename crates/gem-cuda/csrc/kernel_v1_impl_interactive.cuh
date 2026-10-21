// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
#include <cooperative_groups.h>
#include <crates/ulib/includes.hpp>

// Watch formats (passed as flat u32 arrays):
//
// Stop watch: 3 u32s [state_offset, mask, expected]
//   - Stops simulation when (state[offset] & mask) == expected
//
// Toggle watch: 8 u32s [state_offset, mask, start_offset, freq,
//                       posedge_word, posedge_mask, negedge_word, negedge_mask]
//   - Toggles state[offset] ^= mask every freq cycles starting at start_offset
//   - Sets posedge flag on rising edge, negedge flag on falling edge
//   - Use EDGE_FLAG_NONE (0xFFFFFFFF) for posedge_word/negedge_word if not used
//
// Clock edge: 6 u32s [clock_word, clock_mask, pe_word, pe_mask, ne_word,
// ne_mask]
//   - Detects edges by comparing input_state (new) vs output_state (old)
//   - Sets posedge/negedge flags in input_state before simulation
//   - Use EDGE_FLAG_NONE (0xFFFFFFFF) for pe_word/ne_word if not used
//
// Pipe slot layout at dpi_pipe[slot_start]:
//   [0]         status: 0=idle, 1=GPU posted inputs, 2=CPU posted outputs
//   [1..n_in]   input words
//   [n_in+1..]  output words

// Per-cycle DPI call descriptor. Layout must match DpiCallDesc in lib.rs
// (8 x u32). Passed as a flat u32 array and reinterpreted as DpiCallDesc*.
struct DpiCallDesc {
  u32 in_offsets_start; // index into dpi_in_offsets[]
  u32 n_in_words;
  u32 n_in_ports;
  u32 in_ports_start;    // index into dpi_in_port_widths[]
  u32 out_offsets_start; // index into dpi_out_offsets[]
  u32 n_out_ports;
  u32 out_ports_start; // index into dpi_out_port_widths[]
  u32 pipe_slot_start; // u32 offset into dpi_pipe[] for this call's slot
};

// Sentinel value indicating no edge flag
#define EDGE_FLAG_NONE 0xFFFFFFFF

// ============================================================================
// Helpers — all __forceinline__, no cooperative sync inside.
// The caller is responsible for issuing this_grid().sync() around them.
// ============================================================================

// Detect clock edges by comparing new values in state_a against the old values
// still held in state_b. Clears and re-sets posedge/negedge flag bits.
// Called by thread 0 of block 0 only, before the main simulation loop.
__device__ __forceinline__ void detect_clock_edges(const u32 *clock_edges,
                                                   usize num_clock_edges,
                                                   u32 *state_a,
                                                   const u32 *state_b) {
  for (usize i = 0; i < num_clock_edges; ++i) {
    const u32 *ce = clock_edges + i * 6;
    u32 clock_word = ce[0], clock_mask = ce[1];
    u32 pe_word = ce[2], pe_mask = ce[3];
    u32 ne_word = ce[4], ne_mask = ce[5];

    if (pe_word != EDGE_FLAG_NONE)
      state_a[pe_word] &= ~pe_mask;
    if (ne_word != EDGE_FLAG_NONE)
      state_a[ne_word] &= ~ne_mask;

    u32 new_val = state_a[clock_word] & clock_mask;
    u32 old_val = state_b[clock_word] & clock_mask;

    if (old_val == 0 && new_val != 0) {
      if (pe_word != EDGE_FLAG_NONE)
        state_a[pe_word] |= pe_mask;
    } else if (old_val != 0 && new_val == 0) {
      if (ne_word != EDGE_FLAG_NONE)
        state_a[ne_word] |= ne_mask;
    }
  }
}

// Clear edge flags and toggle clocks for one simulation cycle.
// Called by thread 0 of block 0 only.
__device__ __forceinline__ void apply_toggle_watches(usize cycle,
                                                     const u32 *toggle_watches,
                                                     usize num_toggle_watches,
                                                     u32 *input_state,
                                                     u32 *output_state) {
  for (usize w = 0; w < num_toggle_watches; ++w) {
    const u32 *tw = toggle_watches + w * 8;
    u32 tw_offset = tw[0], tw_mask = tw[1];
    u32 tw_start = tw[2], tw_freq = tw[3];
    u32 pe_word = tw[4], pe_mask = tw[5];
    u32 ne_word = tw[6], ne_mask = tw[7];

    if (pe_word != EDGE_FLAG_NONE) {
      input_state[pe_word] &= ~pe_mask;
      output_state[pe_word] &= ~pe_mask;
    }
    if (ne_word != EDGE_FLAG_NONE) {
      input_state[ne_word] &= ~ne_mask;
      output_state[ne_word] &= ~ne_mask;
    }

    if (cycle >= tw_start && (cycle - tw_start) % tw_freq == 0) {
      u32 old_val = input_state[tw_offset] & tw_mask;
      input_state[tw_offset] ^= tw_mask;
      output_state[tw_offset] = input_state[tw_offset];
      u32 new_val = input_state[tw_offset] & tw_mask;

      if (old_val == 0 && new_val != 0) {
        if (pe_word != EDGE_FLAG_NONE) {
          input_state[pe_word] |= pe_mask;
          output_state[pe_word] |= pe_mask;
        }
      } else if (old_val != 0 && new_val == 0) {
        if (ne_word != EDGE_FLAG_NONE) {
          input_state[ne_word] |= ne_mask;
          output_state[ne_word] |= ne_mask;
        }
      }
    }
  }
}

// Check stop watches against output_state. Sets cycles_executed and
// stop_reason if any condition fires.
// Called by thread 0 of block 0 only.
__device__ __forceinline__ void
check_stop_watches(usize cycle, const u32 *stop_watches, usize num_stop_watches,
                   const u32 *output_state, u32 *cycles_executed,
                   u32 *stop_reason) {
  for (usize w = 0; w < num_stop_watches; ++w) {
    const u32 *sw = stop_watches + w * 3;
    if ((output_state[sw[0]] & sw[1]) == sw[2]) {
      *cycles_executed = cycle + 1;
      *stop_reason = 1;
    }
  }
}

// Gather DPI inputs from input_state, hand off to the CPU poll thread via the
// pinned pipe, then scatter the CPU's outputs back into input_state. Executed
// before simulate_block_v1 so the circuit sees DPI outputs as primary inputs
// in the same cycle.
// Called by thread 0 of block 0 only.
__device__ __forceinline__ void dpi_pipe_exchange(
    u32 *dpi_pipe, const u32 *dpi_in_offsets, const u32 *dpi_in_port_widths,
    const u32 *dpi_out_offsets, const u32 *dpi_out_port_widths,
    const DpiCallDesc *dpi_call_descs, usize n_dpi_calls, u32 *input_state) {
  for (usize ci = 0; ci < n_dpi_calls; ++ci) {
    const DpiCallDesc &desc = dpi_call_descs[ci];
    u32 *slot = dpi_pipe + desc.pipe_slot_start;

    for (usize w = 0; w < desc.n_in_words; ++w)
      slot[1 + w] = 0;

    // Write input bits port by port so each port is word-aligned in the slot.
    usize global_bit = 0;
    usize word_offset = 0;
    for (usize p = 0; p < desc.n_in_ports; ++p) {
      u32 port_width = dpi_in_port_widths[desc.in_ports_start + p];
      for (usize b = 0; b < port_width; ++b, ++global_bit) {
        u32 off = dpi_in_offsets[desc.in_offsets_start + global_bit];
        u32 v = (input_state[off >> 5] >> (off & 31)) & 1u;
        slot[1 + word_offset + b / 32] |= v << (b % 32);
      }
      word_offset += (port_width + 31) / 32;
    }

    __threadfence_system();
    atomicExch(&slot[0], 1u);

    while (atomicAdd(&slot[0], 0u) != 2u) {
#if __CUDA_ARCH__ >= 700
      __nanosleep(64u);
#endif
    }
    __threadfence_system();

    // Read output bits port by port; outputs follow the input section in
    // slot.
    global_bit = 0;
    word_offset = 0;
    for (usize p = 0; p < desc.n_out_ports; ++p) {
      u32 port_width = dpi_out_port_widths[desc.out_ports_start + p];
      for (usize b = 0; b < port_width; ++b, ++global_bit) {
        u32 off = dpi_out_offsets[desc.out_offsets_start + global_bit];
        u32 v =
            (slot[1 + desc.n_in_words + word_offset + b / 32] >> (b % 32)) & 1u;
        u32 mask = 1u << (off & 31);
        input_state[off >> 5] =
            (input_state[off >> 5] & ~mask) | (v ? mask : 0u);
      }
      word_offset += (port_width + 31) / 32;
    }

    atomicExch(&slot[0], 0u);
  }
}

// Input and output states are separate buffers
// Stop watches: 3 u32s each [state_offset, mask, expected]
// Toggle watches: 8 u32s each [state_offset, mask, start_offset, freq,
//                              posedge_word, posedge_mask, negedge_word,
//                              negedge_mask]
// Clock edges: 6 u32s each [clock_word, clock_mask, pe_word, pe_mask,
//                            ne_word, ne_mask]
__global__ void simulate_v1_batched_with_watches(
    usize num_blocks, usize num_major_stages,
    const usize *__restrict__ blocks_start, const u32 *__restrict__ blocks_data,
    u32 *__restrict__ sram_data, usize state_size,
    u32 *__restrict__ state_a, // ping-pong buffer A
    u32 *__restrict__ state_b, // ping-pong buffer B
    // Watch parameters
    usize max_cycles,          // run up to N cycles
    const u32 *stop_watches,   // array of stop conditions (3 u32s each)
    usize num_stop_watches,    //
    const u32 *toggle_watches, // array of toggle conditions (8 u32s each)
    usize num_toggle_watches,  //
    // Clock edge detection
    const u32 *clock_edges, // array of clock edge info (6 u32s each)
    usize num_clock_edges,  //
    // Output (in device memory, single write at end)
    u32 *__restrict__ cycles_executed, // how many cycles actually ran
    u32 *__restrict__ stop_reason,     // why we stopped (0=max_cycles, 1=watch
                                       // triggered)
    // DPI pipe - pass dpi_pipe=nullptr and n_dpi_calls=0 to disable
    u32 *__restrict__ dpi_pipe, const u32 *__restrict__ dpi_in_offsets,
    const u32 *__restrict__ dpi_in_port_widths,
    const u32 *__restrict__ dpi_out_offsets,
    const u32 *__restrict__ dpi_out_port_widths,
    const DpiCallDesc *__restrict__ dpi_call_descs, usize n_dpi_calls) {
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 script_starts[32], script_sizes[32];

  // Load per-partition bitstream offsets for each stage
  assert(num_major_stages <= 32);
  if (threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] =
        blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] =
        blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] -
        script_starts[threadIdx.x];
  }
  __syncthreads();

  // Detect clock edges on the freshly uploaded state_a vs the old state_b.
  if (threadIdx.x == 0 && blockIdx.x == 0)
    detect_clock_edges(clock_edges, num_clock_edges, state_a, state_b);
  cooperative_groups::this_grid().sync();

  // Cooperative copy: state_a -> state_b (replaces CPU-side copy)
  usize tid = blockIdx.x * blockDim.x + threadIdx.x;
  usize grid_size = gridDim.x * blockDim.x;
  for (usize i = tid; i < state_size; i += grid_size) {
    state_b[i] = state_a[i];
  }
  cooperative_groups::this_grid().sync();

  u32 *input_state = state_a;
  u32 *output_state = state_b;

  for (usize cycle = 0; cycle < max_cycles; ++cycle) {

    if (threadIdx.x == 0 && blockIdx.x == 0)
      apply_toggle_watches(cycle, toggle_watches, num_toggle_watches,
                           input_state, output_state);
    cooperative_groups::this_grid().sync();

    if (n_dpi_calls > 0 && threadIdx.x == 0 && blockIdx.x == 0)
      // NOTE we currently have one consumer on the CPU so parallelising this
      // would not help. We can think about changing that to turn a sum of all
      // latency into a max of all latencies.
      dpi_pipe_exchange(dpi_pipe, dpi_in_offsets, dpi_in_port_widths,
                        dpi_out_offsets, dpi_out_port_widths, dpi_call_descs,
                        n_dpi_calls, input_state);
    cooperative_groups::this_grid().sync();

    // Run all stages for this cycle
    for (usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(blocks_data + script_starts[stage_i],
                        script_sizes[stage_i], input_state, output_state,
                        sram_data, shared_metadata, shared_writeouts,
                        shared_state);
      // Device-wide sync between RepCut stages (all partitions must finish)
      cooperative_groups::this_grid().sync();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
      check_stop_watches(cycle, stop_watches, num_stop_watches, output_state,
                         cycles_executed, stop_reason);
    cooperative_groups::this_grid().sync();

    bool is_last_cycle = (cycle + 1 >= max_cycles) || *stop_reason;
    if (!is_last_cycle) {
      u32 *tmp = input_state;
      input_state = output_state;
      output_state = tmp;
    }

    if (*stop_reason) {
      break;
    }
  }

  // Normal exit: ran all cycles (only update if no watch triggered)
  if (threadIdx.x == 0 && blockIdx.x == 0 && *stop_reason == 0) {
    *cycles_executed = max_cycles;
  }
}

// Interactive version: executes a single cycle
// Input and output states are separate buffers
__global__ void simulate_v1_interactive_single_cycle(
    usize num_blocks, usize num_major_stages,
    const usize *__restrict__ blocks_start, const u32 *__restrict__ blocks_data,
    u32 *__restrict__ sram_data, usize state_size,
    const u32 *__restrict__ input_state, // Current cycle input state
    u32 *__restrict__ output_state       // Next cycle output state
) {
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  __shared__ u32 script_starts[32], script_sizes[32];

  // Load per-partition bitstream offsets for each stage
  assert(num_major_stages <= 32);
  if (threadIdx.x < num_major_stages) {
    script_starts[threadIdx.x] =
        blocks_start[threadIdx.x * num_blocks + blockIdx.x];
    script_sizes[threadIdx.x] =
        blocks_start[threadIdx.x * num_blocks + blockIdx.x + 1] -
        script_starts[threadIdx.x];
  }
  __syncthreads();

  // Single cycle
  for (usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
    simulate_block_v1(blocks_data + script_starts[stage_i],
                      script_sizes[stage_i], input_state, output_state,
                      sram_data, shared_metadata, shared_writeouts,
                      shared_state);
    // Device-wide sync between RepCut stages (all partitions must finish)
    cooperative_groups::this_grid().sync();
  }
}
