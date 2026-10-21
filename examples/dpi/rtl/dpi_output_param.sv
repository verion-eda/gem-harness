// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Output-parameter DPI example: exercises DPI-C functions with output
// parameters (not just return values).
//
// get_value: void function with input addr and output data.
//            The DPI callback looks up a value by address and writes it
//            to the output parameter.

import "DPI-C" function void get_value(input int addr, output int data);

module dpi_output_param (
    input  logic        clk,
    input  logic [31:0] addr,
    output logic [31:0] data
);
    logic [31:0] dpi_data;

    always @(*)
        get_value(addr, dpi_data);

    always @(posedge clk)
        data <= dpi_data;
endmodule
