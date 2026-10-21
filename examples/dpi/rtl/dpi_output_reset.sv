// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Minimal test case for DPI output parameters with reset-path blocking assigns.
// Exercises the exact SimDRAM pattern: void DPI call with output params in one
// always block, reset-path blocking assignments to those same params in another,
// and registered outputs.  The blocking assigns create dual-driver conflicts.

import "DPI-C" function void tick(input int ctrl, output bit ready, output int data);

module dpi_output_reset (
    input  logic        clk,
    input  logic        reset,
    input  logic [31:0] ctrl,
    output logic        ready_out,
    output logic [31:0] data_out
);
    bit __ready;
    int __data;
    reg __ready_reg;
    reg [31:0] __data_reg;

    // DPI call in its own always block (void call gets replaced by cell inst)
    always @(*)
        tick(ctrl, __ready, __data);

    // Reset path with blocking assigns to DPI output vars → dual-driver conflict
    always @(posedge clk) begin
        if (reset) begin
            __ready = 1'b0;
            __data  = 32'd0;
            __ready_reg <= 1'b0;
            __data_reg  <= 32'd0;
        end else begin
            __ready_reg <= __ready;
            __data_reg  <= __data;
        end
    end

    assign ready_out = __ready_reg;
    assign data_out  = __data_reg;
endmodule
