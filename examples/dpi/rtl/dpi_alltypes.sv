// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// All-types DPI example: exercises every supported DPI-C data type through
// a single module.
//
// identity_byte:     (byte a)     -> byte       (8-bit pass-through)
// identity_shortint: (shortint a) -> shortint   (16-bit pass-through)
// identity_int:      (int a)      -> int        (32-bit pass-through)
// identity_longint:  (longint a)  -> longint    (64-bit pass-through)
// identity_chandle:  (chandle a)  -> chandle    (64-bit opaque handle)
// scale_real:        (real a)     -> real        (64-bit, doubles the input)
// scale_shortreal:   (shortreal a)-> shortreal  (32-bit, doubles the input)
// identity_string:   (string a)   -> string      (64-bit opaque pass-through)
// pack_wide:         (byte a, shortint b, int c) -> longint (multi-arg packing)
// narrow_add:        (shortint x, shortint y) -> byte       (multi-arg narrowing)

import "DPI-C" function byte     identity_byte(input byte a);
import "DPI-C" function shortint identity_shortint(input shortint a);
import "DPI-C" function int      identity_int(input int a);
import "DPI-C" function longint  identity_longint(input longint a);
import "DPI-C" function chandle  identity_chandle(input chandle a);
import "DPI-C" function real     scale_real(input real a);
import "DPI-C" function shortreal scale_shortreal(input shortreal a);
import "DPI-C" function string   identity_string(input string a);
import "DPI-C" function longint  pack_wide(input byte a, input shortint b, input int c);
import "DPI-C" function byte     narrow_add(input shortint x, input shortint y);

module dpi_alltypes (
    input  logic        clk,
    input  logic  [7:0] in_byte,
    input  logic [15:0] in_shortint,
    input  logic [31:0] in_int,
    input  logic [63:0] in_longint,
    input  logic [63:0] in_chandle,
    input  logic [63:0] in_real,
    input  logic [31:0] in_shortreal,
    input  logic [63:0] in_string,
    input  logic  [7:0] pack_a,
    input  logic [15:0] pack_b,
    input  logic [31:0] pack_c,
    input  logic [15:0] narrow_x,
    input  logic [15:0] narrow_y,
    output logic  [7:0] out_byte,
    output logic [15:0] out_shortint,
    output logic [31:0] out_int,
    output logic [63:0] out_longint,
    output logic [63:0] out_chandle,
    output logic [63:0] out_real,
    output logic [31:0] out_shortreal,
    output logic [63:0] out_string,
    output logic [63:0] out_packed,
    output logic  [7:0] out_narrow
);
    always @(posedge clk) begin
        out_byte     <= identity_byte(in_byte);
        out_shortint <= identity_shortint(in_shortint);
        out_int      <= identity_int(in_int);
        out_longint  <= identity_longint(in_longint);
        out_chandle  <= identity_chandle(in_chandle);
        out_real     <= scale_real(in_real);
        out_shortreal <= scale_shortreal(in_shortreal);
        out_string    <= identity_string(in_string);
        out_packed    <= pack_wide(pack_a, pack_b, pack_c);
        out_narrow    <= narrow_add(narrow_x, narrow_y);
    end
endmodule
