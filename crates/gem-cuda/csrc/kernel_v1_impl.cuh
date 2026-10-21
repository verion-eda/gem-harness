// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//
// GEM Virtual Boolean Processor kernel - interprets VLIW bitstream to simulate
// partitioned E-AIG circuits. Each thread block = 1 Boolean core (partition).
// State: 256 threads × 32 bits = 8192 bits per partition.

#include <cooperative_groups.h>
#include <crates/ulib/includes.hpp>

#ifndef __VECTOR_READ__
#define __VECTOR_READ__

// Aligned vector types for coalesced global memory reads
struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) { *this = *t; }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) { *this = *t; }
};

#endif

// Executes one partition's bitstream for one RepCut stage.
// shared_state[256]: 8192-bit local state vector
// shared_writeouts[256]: output staging for partition I/O
// shared_metadata[256]: instruction metadata buffer
__device__ void simulate_block_v1(
    const u32 *__restrict__ script, // VLIW bitstream
    usize script_size,
    const u32 *__restrict__ input_state, // global state (cycle N)
    u32 *__restrict__ output_state,      // global state (cycle N+1)
    u32 *__restrict__ sram_data, // RAM blocks (13-bit addr, 32-bit data)
    u32 *__restrict__ shared_metadata, u32 *__restrict__ shared_writeouts,
    u32 *__restrict__ shared_state) // __restrict__ is a compiler hint saying
                                    // "this pointer doesn't alias (overlap)
                                    // with other pointers", opening paths for
                                    // optimisation
{
  int script_pi = 0; // bitstream instruction pointer
  while (true) {
    VectorRead2 t2_1, t2_2;
    VectorRead4 t4_1, t4_2, t4_3, t4_4, t4_5;

    // === INIT INSTRUCTION (8192-bit) ===
    // Load partition metadata: #layers, I/O config, RAM offsets
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    script_pi += 256;
    t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
    __syncthreads();

    int num_stages = shared_metadata[0]; // # boomerang layers
    if (!num_stages) {
      break;
    }
    int is_last_part = shared_metadata[1];
    int num_ios = shared_metadata[2];     // # I/O ports
    int io_offset = shared_metadata[3];   // global state offset
    int num_srams = shared_metadata[4];   // # RAM blocks
    int sram_offset = shared_metadata[5]; // RAM base address
    int num_global_read_rounds = shared_metadata[6];
    int num_output_duplicates = shared_metadata[7];

    // Unpack per-thread writeout hook (2 × 16-bit indices packed per word)
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if (threadIdx.x % 2 == 0) // even threadIdx
    {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    } else // odd threadIdx
    {
      writeout_hook_i = writeout_hook_i >> 16;
    }

    // Prefetch first boomerang layer's permutation data
    t4_1.read((const VectorRead4 *)(script + script_pi +
                                    256 * 2 * num_global_read_rounds) +
              threadIdx.x);
    t4_2.read((const VectorRead4 *)(script + script_pi +
                                    256 * 2 * num_global_read_rounds +
                                    256 * 4) +
              threadIdx.x);
    t4_3.read((const VectorRead4 *)(script + script_pi +
                                    256 * 2 * num_global_read_rounds +
                                    256 * 4 * 2) +
              threadIdx.x);
    t4_4.read((const VectorRead4 *)(script + script_pi +
                                    256 * 2 * num_global_read_rounds +
                                    256 * 4 * 3) +
              threadIdx.x);
    t4_5.read((const VectorRead4 *)(script + script_pi +
                                    256 * 2 * num_global_read_rounds +
                                    256 * 4 * 4) +
              threadIdx.x);

    // === GLOBAL STATE READ (16384-bit instructions) ===
    // Load partition inputs from global memory. Each round:
    //   idx = global memory offset, mask = which bits to extract
    //
    // TODO We should benchmark the numbor of read rounds to understand if we
    // can minimize the rounds with a better placement algorithm (layers)
    u32 t_global_rd_state = 0;
    for (int gr_i = 0; gr_i < num_global_read_rounds; gr_i += 2) {
      u32 idx = t2_1.c1;
      u32 mask = t2_1.c2;
      script_pi += 256 * 2;
      t2_2.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if (mask) {
        // MSB of idx selects input_state vs output_state (for feedback paths)
        const u32 *real_input_array;
        if (idx >> 31)
          real_input_array = output_state - (1 << 31);
        else
          real_input_array = input_state;

        // XXX this is not memory coalescing efficient (wrap loading more cache
        // lines than minimum theoretically needed) due to graph layout. We see
        // this behavior in many places where we access input/output arrays but
        // this is mostly happening at the boundary and not in hot loops.
        u32 value = real_input_array[idx];

        // Extract masked bits into t_global_rd_state (LSB-first)
        while (mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if (value & lowbit)
            t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }

      if (gr_i + 1 >= num_global_read_rounds)
        break;
      idx = t2_2.c1;
      mask = t2_2.c2;
      script_pi += 256 * 2;
      t2_1.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      if (mask) {
        const u32 *real_input_array;
        if (idx >> 31)
          real_input_array = output_state - (1 << 31);
        else
          real_input_array = input_state;
        u32 value = real_input_array[idx];
        while (mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if (value & lowbit)
            t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    __syncthreads();

    // === BOOMERANG LAYER LOOP ===
    // Each layer folds 8192 bits -> 1 bit using:
    //
    //   result = (A ^ XOR.A) & ((B ^ XOR.B) | OR.B)
    //
    // - the `XOR` constant just encodes the presents of the `NOT` in the graph
    // - the trailing `| OR.B` encodes the "is vacant" position: `OR.B = 1`
    //   forces B' to 1 regardless of what B actually is -> `result = (A ^
    //   XOR.A)`
    for (int bs_i = 0; bs_i < num_stages; ++bs_i) {
      u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0,
          hier_flag_orb = 0;

// === LOCAL BIT PERMUTATION (32768-bit instruction) ===
// Gather 16 input bits per thread from arbitrary shared_state locations.
// t_shuffle encodes two 16-bit indices: [idx2:idx1]
//   idx >> 5 = word slot (0-255), idx & 31 = bit within word
#define GEMV1_SHUF_INPUT_K(k_outer, k_inner, t_shuffle)                        \
  {                                                                            \
    u32 k = k_outer * 4 + k_inner;                                             \
    u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);                         \
    u32 t_shuffle_2_idx = t_shuffle >> 16;                                     \
                                                                               \
    hier_input |=                                                              \
        (shared_state[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1)     \
        << (k * 2);                                                            \
    hier_input |=                                                              \
        (shared_state[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1)     \
        << (k * 2 + 1);                                                        \
  }
#define GEMV1_SHUF_INPUT_K_4(k_outer, t_shuffle)                               \
  {                                                                            \
    GEMV1_SHUF_INPUT_K(k_outer, 0, t_shuffle.c1);                              \
    GEMV1_SHUF_INPUT_K(k_outer, 1, t_shuffle.c2);                              \
    GEMV1_SHUF_INPUT_K(k_outer, 2, t_shuffle.c3);                              \
    GEMV1_SHUF_INPUT_K(k_outer, 3, t_shuffle.c4);                              \
  }
      script_pi += 256 * 4 * 5;

      // XXX here (and other places where we access the shared memory) we get a
      // lot of shared memory bank conflicts there is not much we can do with
      // "random" gathers across 256 shared memory words from threads.
      //
      // The paper states this in Observation 2: irregular access inside the
      // GPU shared memory is much less costly than irregular global memory
      // access
      GEMV1_SHUF_INPUT_K_4(0, t4_1);
      t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(1, t4_2);
      t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) +
                threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(2, t4_3);
      t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) +
                threadIdx.x);
      GEMV1_SHUF_INPUT_K_4(3, t4_4);
      t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) +
                threadIdx.x);
#undef GEMV1_SHUF_INPUT_K
#undef GEMV1_SHUF_INPUT_K_4

      // Fold masks from instruction: control inversion and bypass
      hier_flag_xora = t4_5.c1; // XOR.A: invert input A
      hier_flag_xorb = t4_5.c2; // XOR.B: invert input B
      hier_flag_orb = t4_5.c3;  // OR.B:  bypass B (force to 1)
      t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) +
                threadIdx.x);

      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      __syncthreads();

      // === HIERARCHICAL FOLD (boomerang) ===
      // The hierarchical fold is using a "reduction" pattern using different
      // thread ids for different levels. The implementations partitions the
      // fold into 3 phases (reduction in levels 1,2,3):
      //   - level 0: 128 threads
      //   - level 1: 64 threads
      //   - level 2: 32 threads
      //   - level 3: 16 threads
      //   - level 4,5,6,7: 32 threads switching to warp shuffle (no shared
      //   memory access)
      //   - rest is done on thread id 0 using bit masks, no parallelism
      //
      // hier[0]: 8192 -> 4096 bits (threads 128-255 active)
      if (threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 ret = (hier_input_a ^ hier_flag_xora) &
                  ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
        shared_state[threadIdx.x] = ret;
      }
      __syncthreads();

      // hier[1..3]: 4096 -> 512 bits (threads 64 - 127 active)
      u32 tmp_cur_hi;
      for (int hi = 1; hi <= 3; ++hi) {
        int hier_width = 1 << (7 - hi);
        if (threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 ret = (hier_input_a ^ hier_flag_xora) &
                    ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          tmp_cur_hi = ret;
          shared_state[threadIdx.x] = ret;
        }
        __syncthreads();
      }

      // hier[4..7]: 512 -> 32 bits (warp shuffles - no sync needed using
      // __shfl_down_sync)
      // threads 0 - 31
      if (threadIdx.x < 32) {
        for (int hi = 4; hi <= 7; ++hi) {
          int hier_width = 1 << (7 - hi);
          u32 hier_input_a =
              __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width);
          u32 hier_input_b =
              __shfl_down_sync(0xffffffff, tmp_cur_hi, hier_width * 2);
          if (threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
            tmp_cur_hi = (hier_input_a ^ hier_flag_xora) &
                         ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          }
        }
        u32 v1 = __shfl_down_sync(0xffffffff, tmp_cur_hi, 1);

        // hier[8..12]: 32 -> 1 bit (single thread, bit manipulation)
        // only thread 0, parallelism does not makes sense here
        if (threadIdx.x == 0) {
          u32 r8 = ((v1 << 16) ^ hier_flag_xora) &
                   ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
          u32 r9 = ((r8 >> 8) ^ hier_flag_xora) &
                   (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
          u32 r10 = ((r9 >> 4) ^ hier_flag_xora) &
                    (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
          u32 r11 = ((r10 >> 2) ^ hier_flag_xora) &
                    (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) &
                    12 /* 0b1100 */;
          u32 r12 = ((r11 >> 1) ^ hier_flag_xora) &
                    (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) &
                    2 /* 0b10 */;
          tmp_cur_hi = r8 | r9 | r10 | r11 | r12;
        }
        shared_state[threadIdx.x] = tmp_cur_hi;
      }
      __syncthreads();

      // Capture layer outputs at designated hook points
      if ((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
      }
    }
    __syncthreads();

    // === SRAM & OUTPUT PERMUTATION ===
    // Gather bits for RAM addresses and output duplication
    u32 sram_duplicate_t = 0;
#define GEMV1_SHUF_SRAM_DUPL_K(k_outer, k_inner, t_shuffle)                    \
  {                                                                            \
    u32 k = k_outer * 4 + k_inner;                                             \
    u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);                         \
    u32 t_shuffle_2_idx = t_shuffle >> 16;                                     \
                                                                               \
    sram_duplicate_t |=                                                        \
        (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) \
        << (k * 2);                                                            \
    sram_duplicate_t |=                                                        \
        (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) \
        << (k * 2 + 1);                                                        \
  }
#define GEMV1_SHUF_SRAM_DUPL_K_4(k_outer, t_shuffle)                           \
  {                                                                            \
    GEMV1_SHUF_SRAM_DUPL_K(k_outer, 0, t_shuffle.c1);                          \
    GEMV1_SHUF_SRAM_DUPL_K(k_outer, 1, t_shuffle.c2);                          \
    GEMV1_SHUF_SRAM_DUPL_K(k_outer, 2, t_shuffle.c3);                          \
    GEMV1_SHUF_SRAM_DUPL_K(k_outer, 3, t_shuffle.c4);                          \
  }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_SRAM_DUPL_K_4(0, t4_1);
    t4_1.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(1, t4_2);
    t4_2.read(((const VectorRead4 *)(script + script_pi + 256 * 4)) +
              threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(2, t4_3);
    t4_3.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 2)) +
              threadIdx.x);
    GEMV1_SHUF_SRAM_DUPL_K_4(3, t4_4);
    t4_4.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 3)) +
              threadIdx.x);
#undef GEMV1_SHUF_SRAM_DUPL_K_4
#undef GEMV1_SHUF_SRAM_DUPL_K
    // same XOR trick to invert or bypass
    sram_duplicate_t = (sram_duplicate_t & ~t4_5.c2) ^ t4_5.c1;
    t4_5.read(((const VectorRead4 *)(script + script_pi + 256 * 4 * 4)) +
              threadIdx.x);

    // === SRAM READ/WRITE ===
    // Each SRAM: 8KB (13-bit addr, 32-bit data). 4 threads cooperate per SRAM.
    // Thread layout per SRAM: [addr] [wr_en] [wr_data] [unused]
    u32 *ram = nullptr;
    u32 r, w0;
    u32 port_w_addr_iv, port_w_wr_en, port_w_wr_data_iv;
    if (threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t; // [wr_addr:rd_addr] packed
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
                     ? 0xffffffff
                     : (0xffffffff >> (last_tid - num_srams * 4));
      // Warp shuffle to distribute wr_en and wr_data from neighboring threads
      port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);

      if (threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13); // 8KB per SRAM
        u32 port_r_addr_iv = addrs & 0xffff;
        port_w_addr_iv = addrs >> 16;

        ram = sram_data + sram_st;
        r = ram[port_r_addr_iv];  // synchronous read
        w0 = ram[port_w_addr_iv]; // read-before-write for masked update
      }
    }
    // __syncthreads();

    // === CLOCK ENABLE PERMUTATION ===
    // Gather per-bit clock enable signals for FF updates
    u32 clken_perm = 0;
#define GEMV1_SHUF_CLKEN_K(k_outer, k_inner, t_shuffle)                        \
  {                                                                            \
    u32 k = k_outer * 4 + k_inner;                                             \
    u32 t_shuffle_1_idx = t_shuffle & ((1 << 16) - 1);                         \
    u32 t_shuffle_2_idx = t_shuffle >> 16;                                     \
                                                                               \
    clken_perm |=                                                              \
        (shared_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31) & 1) \
        << (k * 2);                                                            \
    clken_perm |=                                                              \
        (shared_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31) & 1) \
        << (k * 2 + 1);                                                        \
  }
#define GEMV1_SHUF_CLKEN_K_4(k_outer, t_shuffle)                               \
  {                                                                            \
    GEMV1_SHUF_CLKEN_K(k_outer, 0, t_shuffle.c1);                              \
    GEMV1_SHUF_CLKEN_K(k_outer, 1, t_shuffle.c2);                              \
    GEMV1_SHUF_CLKEN_K(k_outer, 2, t_shuffle.c3);                              \
    GEMV1_SHUF_CLKEN_K(k_outer, 3, t_shuffle.c4);                              \
  }
    script_pi += 256 * 4 * 5;
    GEMV1_SHUF_CLKEN_K_4(0, t4_1);
    GEMV1_SHUF_CLKEN_K_4(1, t4_2);
    GEMV1_SHUF_CLKEN_K_4(2, t4_3);
    GEMV1_SHUF_CLKEN_K_4(3, t4_4);
#undef GEMV1_SHUF_CLKEN_K
#undef GEMV1_SHUF_CLKEN_K_4

    // === SRAM COMMIT ===
    // Write SRAM read results to outputs, perform masked write-back
    // XXX Note that reading/writing into ram[..] is inherentlyu non-coalesced
    // and unpredictable. This is the worst meemory access pattern here.
    if (threadIdx.x < num_srams * 4) {
      if (threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        // Masked write: (old & ~mask) | (new & mask)
        ram[port_w_addr_iv] =
            (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
      }
    } else if (threadIdx.x < num_srams * 4 + num_output_duplicates) {
      shared_writeouts[num_ios - num_srams - num_output_duplicates +
                       (threadIdx.x - num_srams * 4)] = sram_duplicate_t;
    }

    __syncthreads();
    u32 writeout_inv = shared_writeouts[threadIdx.x];

    // Apply final masks
    clken_perm = (clken_perm & ~t4_5.c2) ^ t4_5.c1;
    writeout_inv ^= t4_5.c3; // optional output inversion

    // === GLOBAL OUTPUT COMMIT ===
    // Clock-gated state update: only bits with clken=1 are updated
    if (threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;
    }

    if (is_last_part)
      break;
  }
  assert(script_size == script_pi);
}

// Top-level kernel: runs multi-cycle simulation with RepCut stage
// synchronization. Grid: num_blocks (1 block per partition), Block: 256 threads
__global__ void simulate_v1_noninteractive_simple_scan(
    usize num_blocks,                       // # partitions
    usize num_major_stages,                 // # RepCut stages (horizontal cuts)
    const usize *__restrict__ blocks_start, // bitstream offsets [stage *
                                            // num_blocks + block]
    const u32 *__restrict__ blocks_data,    // concatenated bitstreams
    u32 *__restrict__ sram_data, // RAM blocks (persistent across cycles)
    usize num_cycles,
    usize state_size,                       // words per cycle state
    u32 *__restrict__ states_noninteractive // state buffer [(num_cycles+1) *
                                            // state_size]
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

  // Main simulation loop: cycles × stages
  for (usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    for (usize stage_i = 0; stage_i < num_major_stages; ++stage_i) {
      simulate_block_v1(
          blocks_data + script_starts[stage_i], script_sizes[stage_i],
          states_noninteractive + cycle_i * state_size,
          states_noninteractive + (cycle_i + 1) * state_size, sram_data,
          shared_metadata, shared_writeouts, shared_state);
      // Device-wide sync between RepCut stages (all partitions must finish)
      cooperative_groups::this_grid().sync();
    }
  }
}
