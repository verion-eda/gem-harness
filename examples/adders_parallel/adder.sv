// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// 32-bit combinational adder (building block)
module adder_32(
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] sum,
    output wire        carry
  );
  assign {carry, sum} = a + b;
endmodule

// Top-level: N parallel synchronous 32-bit adders for scaling tests
module adders_parallel #(
    parameter N = 1024
  )(
    input  wire              clk,
    input  wire              rst,
    input  wire [N*32-1:0]   a,
    input  wire [N*32-1:0]   b,
    output wire [N*32-1:0]   sum,
    output wire [N-1:0]      carry
  );
  localparam WIDTH = 32;
  wire [N*32-1:0] sum_comb;
  wire [N-1:0] carry_comb;

  // N parallel 32-bit adders (combinational)
  genvar i;
  generate
    for (i = 0; i < N; i = i + 1)
    begin : adder_gen
      adder_32 u_adder (
                 .a    (a[WIDTH*i +: WIDTH]),
                 .b    (b[WIDTH*i +: WIDTH]),
                 .sum  (sum_comb[WIDTH*i +: WIDTH]),
                 .carry(carry_comb[i])
               );
    end
  endgenerate

  // Synchronous output registers
  reg [N*32-1:0] sum_reg;
  reg [N-1:0] carry_reg;
  always_ff @(posedge clk)
  begin
    if (rst)
    begin
      sum_reg   <= '0;
      carry_reg <= '0;
    end
    else
    begin
      sum_reg   <= sum_comb;
      carry_reg <= carry_comb;
    end
  end

  assign sum   = sum_reg;
  assign carry = carry_reg;
endmodule
