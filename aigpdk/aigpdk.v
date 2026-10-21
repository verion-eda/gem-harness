// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.

// Notation: AND2_<inverted_input_0>_<inverted_input_1>_<output_inverted>

module AND2_00_0 (A, B, Y);
  input  A ;
  input  B ;
  output Y ;

  assign Y = A & B;

endmodule // AND2_00_0

module AND2_01_0 (A, B, Y);
  input  A ;
  input  B ;
  output Y ;

  assign Y = A & ~B;

endmodule // AND2_01_0

module AND2_10_0 (A, B, Y);
  input  A ;
  input  B ;
  output Y ;

  assign Y = ~A & B;

endmodule // AND2_10_0

module AND2_11_0 (A, B, Y);
  input  A ;
  input  B ;
  output Y ;

  assign Y = ~A & ~B;

endmodule // AND2_11_0

module AND2_11_1 (A, B, Y);
  input  A ;
  input  B ;
  output Y ;

  assign Y = A | B;

endmodule // AND2_11_1

module INV (A, Y);
  input  A ;
  output Y ;

  not (Y, A);

endmodule // INV

module BUF (A, Y);
  input  A ;
  output Y ;

  assign Y = A;

endmodule // BUF

// module LATCH (CLK, D, Q);
// input  CLK ;
// input  D ;
// output Q ;
// reg NOTIFIER ;

//    udp_tlat (DS0000, D, CLK, 1'B0, 1'B0, NOTIFIER);
//    not (P0000, DS0000);
//    buf (Q, DS0000);

// endmodule

module CKLNQD (CP, E, Q);
  (* gated_clock = "true" *) input CP;
  input E;
  output Q;
  reg    QD;
  always @*
  begin
    if (~CP)
      QD <= E;
  end
  assign Q = CP & QD;
endmodule
