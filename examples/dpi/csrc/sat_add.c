// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Saturating 32-bit signed add — C implementation of the DPI function.
//
// Inputs:  word 0 = a (int32), word 1 = b (int32)
// Outputs: word 0 = saturating_add(a, b) (int32)
//
// See `SimulatorBuilder::register_dpi_lib()` for the calling convention.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

void harness_dpi_sat_add(const uint32_t* inputs, size_t n_in,
                        uint32_t* outputs, size_t n_out) {
    assert(n_in == 2 && "sat_add expects 2 input words (a, b)");
    assert(n_out == 1 && "sat_add expects 1 output word");

    int32_t a = (int32_t)inputs[0];
    int32_t b = (int32_t)inputs[1];

    // Saturating add: clamp to INT32_MIN / INT32_MAX on overflow.
    int64_t sum = (int64_t)a + (int64_t)b;
    int32_t result;
    if (sum > INT32_MAX)
        result = INT32_MAX;
    else if (sum < INT32_MIN)
        result = INT32_MIN;
    else
        result = (int32_t)sum;

    outputs[0] = (uint32_t)result;
}
