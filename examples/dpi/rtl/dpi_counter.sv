// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Original SV with DPI: a saturating accumulator.
// This file is for reference only; synthesis uses dpi_counter_synth.sv.
import "DPI-C" function int sat_add(input int a, input int b);
import "DPI-C" function void trace_event(input int x);

module dpi_counter (
    input  logic        clk,
    input  logic [31:0] inc,
    output logic [31:0] count
);
    always @(posedge clk)
        count <= sat_add(count, inc);

    always @(posedge clk)
        trace_event(count);
endmodule
