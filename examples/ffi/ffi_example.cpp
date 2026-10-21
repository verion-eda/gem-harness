// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
/*
 * Example C++ program using gem-harness Interactive Simulator FFI
 * Demonstrates runtime state validation at the FFI boundary
 *
 * This example shows how the FFI uses runtime state checking in C,
 * while the Rust side can optionally use compile-time typestate enforcement.
 *
 * Compile with:
 *   g++ examples/ffi/ffi_example.cpp -L./target/release -lsimulator -o ffi_example
 */

#include "harness.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK_STATE(res, expected_state, operation)                            \
  do {                                                                         \
    if ((res).state == SimState::Error) {                                      \
      fprintf(stderr, "ERROR in %s: %s\n", operation,                          \
              (res).error_msg ? (res).error_msg : "unknown error");            \
      exit(1);                                                                 \
    }                                                                          \
    if ((res).state != expected_state) {                                       \
      fprintf(stderr, "ERROR in %s: expected state %d, got state %d\n",        \
              operation, expected_state, (res).state);                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <netlist.v> <parts.gem>\n", argv[0]);
    return 1;
  }

  const char *netlist_path = argv[1];
  const char *gemparts_path = argv[2];

  SimResult res = sim_create(netlist_path, gemparts_path, 1, 0);
  CHECK_STATE(res, SimState::Initialized, "sim_create");
  printf("Created simulator (state = Initialized)\n");

  res = sim_reset(res.handle);
  CHECK_STATE(res, SimState::Ready, "sim_reset");
  printf("Reset complete (state = Ready)\n");

  printf("\nSimulator info:\n");
  printf("  Inputs:  %zu\n", sim_num_inputs(res.handle));
  printf("  Outputs: %zu\n", sim_num_outputs(res.handle));

  // Setup tracing (works in any state)
  const char *trace_signals[] = {"sum", "carry"};
  SimResult trace_res = sim_trace_signals(res.handle, trace_signals, 2);
  if (trace_res.state == SimState::Error) {
    fprintf(stderr, "Warning: trace setup failed: %s\n", trace_res.error_msg);
  } else {
    printf("\nTracing signals: sum, carry\n");
  }

  for (int cycle = 0; cycle < 2; cycle++) {
    bool a_val = (cycle % 2) == 0;
    bool b_val = !a_val;

    res = sim_poke_signal(res.handle, "a", a_val);
    CHECK_STATE(res, SimState::Poked, "sim_poke_signal a");
    printf("Poked a = %d (state = Poked)\n", a_val);

    res = sim_poke_signal(res.handle, "b", b_val);
    CHECK_STATE(res, SimState::Poked, "sim_poke_signal b");
    printf("Poked b = %d (state = Poked)\n", b_val);

    res = sim_step(res.handle);
    CHECK_STATE(res, SimState::Stepped, "sim_step");
    printf("Stepped (state = Stepped)\n");

    uint64_t sum, carry;
    res = sim_peek_signal(res.handle, "sum", &sum);
    CHECK_STATE(res, SimState::Stepped, "sim_peek_signal sum");

    res = sim_peek_signal(res.handle, "carry", &carry);
    CHECK_STATE(res, SimState::Stepped, "sim_peek_signal carry");

    printf("Results: sum = %llu, carry = %llu\n", (unsigned long long)sum,
           (unsigned long long)carry);

    res = sim_next_cycle(res.handle);
    CHECK_STATE(res, SimState::Poked, "sim_next_cycle");
  }

  // Dump VCD trace (works in any state)
  printf("\n=== Dumping VCD trace ===\n");
  SimResult dump_res = sim_dump(res.handle, "output.vcd");
  if (dump_res.state == SimState::Error) {
    fprintf(stderr, "Warning: VCD dump failed: %s\n", dump_res.error_msg);
  } else {
    printf("Traces written to output.vcd\n");
  }

  sim_destroy(res.handle);
  printf("\nSimulator destroyed\n");

  return 0;
}

/*
 * Example with direct AIG pin access (lower level, faster):
 */
void example_with_aigpins() {
  SimResult res = sim_create("netlist.v", "parts.gem", 1, 0);
  res = sim_reset(res.handle);

  // Assuming we know the AIG pin numbers from prior analysis
  const size_t INPUT_PIN_A = 42;
  const size_t INPUT_PIN_B = 43;
  const size_t OUTPUT_PIN_Z = 100;

  for (int cycle = 0; cycle < 5; cycle++) {
    // Poke by AIG pin number (faster, no name lookup)
    res = sim_poke(res.handle, INPUT_PIN_A, true);
    if (res.state == SimState::Error)
      break;

    res = sim_poke(res.handle, INPUT_PIN_B, false);
    if (res.state == SimState::Error)
      break;

    res = sim_step(res.handle);
    if (res.state == SimState::Error)
      break;

    bool output;
    res = sim_peek(res.handle, OUTPUT_PIN_Z, &output);
    if (res.state == SimState::Error)
      break;

    printf("Cycle %d: output = %d\n", cycle, output);

    res = sim_next_cycle(res.handle);
    if (res.state == SimState::Error)
      break;
  }

  sim_destroy(res.handle);
}
