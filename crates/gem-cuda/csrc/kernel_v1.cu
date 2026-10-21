// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.

#include <cstdio>
#include "kernel_v1_impl.cuh"
#include "kernel_v1_impl_interactive.cuh"

#define checkCudaErrors(call)                                                  \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

extern "C" void simulate_v1_noninteractive_simple_scan_cuda(
    usize num_blocks, usize num_major_stages, const usize *blocks_start,
    const u32 *blocks_data, u32 *sram_data, usize num_cycles, usize state_size,
    u32 *states_noninteractive) {
  void *arg_ptrs[8] = {(void *)&num_blocks,   (void *)&num_major_stages,
                       (void *)&blocks_start, (void *)&blocks_data,
                       (void *)&sram_data,    (void *)&num_cycles,
                       (void *)&state_size,   (void *)&states_noninteractive};
  checkCudaErrors(cudaLaunchCooperativeKernel(
      (void *)simulate_v1_noninteractive_simple_scan, num_blocks, 256, arg_ptrs,
      0, (cudaStream_t)0));
}

extern "C" void simulate_v1_interactive_single_cycle_cuda(
    usize num_blocks, usize num_major_stages, const usize *blocks_start,
    const u32 *blocks_data, u32 *sram_data, usize state_size,
    const u32 *input_state, u32 *output_state) {
  void *arg_ptrs[8] = {(void *)&num_blocks,   (void *)&num_major_stages,
                       (void *)&blocks_start, (void *)&blocks_data,
                       (void *)&sram_data,    (void *)&state_size,
                       (void *)&input_state,  (void *)&output_state};
  checkCudaErrors(cudaLaunchCooperativeKernel(
      (void *)simulate_v1_interactive_single_cycle, num_blocks, 256, arg_ptrs,
      0, (cudaStream_t)0));
}

// XXX needs _cuda in the name so that we get a binding that runs on the device
extern "C" void simulate_v1_batches_with_watches_cuda(
    usize num_blocks, usize num_major_stages, const usize *blocks_start,
    const u32 *blocks_data, u32 *sram_data, usize state_size, u32 *state_a,
    u32 *state_b, usize max_cycles, const u32 *stop_watches,
    usize num_stop_watches, const u32 *toggle_watches, usize num_toggle_watches,
    const u32 *clock_edges, usize num_clock_edges, u32 *cycles_retired,
    u32 *stop_reason, u32 *__restrict__ dpi_pipe,
    const u32 *__restrict__ dpi_in_offsets,
    const u32 *__restrict__ dpi_in_port_widths,
    const u32 *__restrict__ dpi_out_offsets,
    const u32 *__restrict__ dpi_out_port_widths,
    const DpiCallDesc *__restrict__ dpi_call_descs, usize n_dpi_calls) {
  void *arg_ptrs[24] = {(void *)&num_blocks,
                        (void *)&num_major_stages,
                        (void *)&blocks_start,
                        (void *)&blocks_data,
                        (void *)&sram_data,
                        (void *)&state_size,
                        (void *)&state_a,
                        (void *)&state_b,
                        (void *)&max_cycles,
                        (void *)&stop_watches,
                        (void *)&num_stop_watches,
                        (void *)&toggle_watches,
                        (void *)&num_toggle_watches,
                        (void *)&clock_edges,
                        (void *)&num_clock_edges,
                        (void *)&cycles_retired,
                        (void *)&stop_reason,
                        (void *)&dpi_pipe,
                        (void *)&dpi_in_offsets,
                        (void *)&dpi_in_port_widths,
                        (void *)&dpi_out_offsets,
                        (void *)&dpi_out_port_widths,
                        (void *)&dpi_call_descs,
                        (void *)&n_dpi_calls};

  checkCudaErrors(cudaLaunchCooperativeKernel(
      (void *)simulate_v1_batched_with_watches, num_blocks, 256, arg_ptrs, 0,
      (cudaStream_t)0));
  checkCudaErrors(cudaDeviceSynchronize());
}
