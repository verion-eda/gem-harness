// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI unpacked-array example: exercises byte[] parameters (input and output).
//
// array_tick: void function with both input and output unpacked byte arrays,
//             plus a scalar control signal.  Mimics the SimDRAM memory_tick
//             pattern where data buses are passed as byte arrays.

import "DPI-C" function void array_tick(
    input  int   ctrl,
    input  byte  w_data[],
    output byte  r_data[]
);

module dpi_array #(
    parameter DATA_BITS = 64,
    parameter BYTES     = DATA_BITS / 8
)(
    input                   clk,
    input  logic [31:0]     ctrl,
    input  logic [DATA_BITS-1:0] data_in,
    output logic [DATA_BITS-1:0] data_out
);

    byte __w_data[(DATA_BITS / 8)-1:0];
    byte __r_data[(DATA_BITS / 8)-1:0];

    reg [DATA_BITS-1:0] data_out_reg;

    // Pack input: scalar vector -> unpacked byte array
    generate
        for (genvar i = 0; i < DATA_BITS / 8; i++) begin : gen_pack
            assign __w_data[i] = data_in[i * 8 +: 8];
        end
    endgenerate

    // DPI call in its own always block (void call gets replaced by cell inst)
    always @(posedge clk) begin
        array_tick(ctrl, __w_data, __r_data);
    end

    // Unpack output: unpacked byte array -> scalar vector
    generate
        for (genvar i = 0; i < DATA_BITS / 8; i++) begin : gen_unpack
            assign data_out[i * 8 +: 8] = __r_data[i];
        end
    endgenerate

endmodule
