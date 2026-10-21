// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI example with constant inputs: tests that constant parameters
// are correctly resolved at build time (not via runtime AIG traversal).
import "DPI-C" function int const_add(input int a, input int b);

module dpi_const_input (
    input  logic        clk,
    output logic [31:0] result
);
    // 'a' is a constant 42, 'b' is a constant 7.
    // The DPI function should receive (42, 7) every cycle.
    always @(posedge clk)
        result <= const_add(32'd42, 32'd7);
endmodule
