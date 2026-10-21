// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Duplicate of sat_add.c — used only to test that loading two libraries
// exporting the same symbol is rejected.

#include <stdint.h>
#include <stddef.h>

void harness_dpi_sat_add(const uint32_t* inputs, size_t n_in,
                        uint32_t* outputs, size_t n_out) {
    (void)inputs; (void)n_in; (void)n_out;
    outputs[0] = 0;
}
