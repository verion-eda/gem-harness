// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
module alu #(
    parameter WIDTH = 32
)(
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire [3:0]       op,
    output reg  [WIDTH-1:0] result,
    output reg              zero,
    output reg              carry,
    output reg              overflow,
    output reg              sign
);

    // opcodes
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_NOR  = 4'b0101;
    localparam OP_SLL  = 4'b0110; // shift left logical
    localparam OP_SRL  = 4'b0111; // shift right logical
    localparam OP_SRA  = 4'b1000; // shift right arithmetic
    localparam OP_SLT  = 4'b1001; // set less than (signed)
    localparam OP_SLTU = 4'b1010; // set less than (unsigned)
    localparam OP_PASS = 4'b1011; // pass-through a

    reg [WIDTH:0] addsub; // for overflow/carry check

    always @* begin
        addsub = {WIDTH+1{1'b0}}; // default assignment to avoid latch
        case (op)
            OP_ADD: begin
                addsub = {1'b0, a} + {1'b0, b};
                result = addsub[WIDTH-1:0];
                carry = addsub[WIDTH];
                overflow = (~a[WIDTH-1] & ~b[WIDTH-1] & result[WIDTH-1])
                        | ( a[WIDTH-1] &  b[WIDTH-1] & ~result[WIDTH-1]);
            end
            OP_SUB: begin
                addsub = {1'b0, a} - {1'b0, b};
                result = addsub[WIDTH-1:0];
                carry = addsub[WIDTH];
                // subtraction overflow: (a[31] != b[31]) && (result[31] != a[31])
                overflow = (a[WIDTH-1] ^ b[WIDTH-1]) && (result[WIDTH-1] ^ a[WIDTH-1]);
            end
            OP_AND: begin
                result = a & b;
                carry = 0;
                overflow = 0;
            end
            OP_OR: begin
                result = a | b;
                carry = 0;
                overflow = 0;
            end
            OP_XOR: begin
                result = a ^ b;
                carry = 0;
                overflow = 0;
            end
            OP_NOR: begin
                result = ~(a | b);
                carry = 0;
                overflow = 0;
            end
            OP_SLL: begin
                result = a << b[$clog2(WIDTH)-1:0];
                carry = 0;
                overflow = 0;
            end
            OP_SRL: begin
                result = a >> b[$clog2(WIDTH)-1:0];
                carry = 0;
                overflow = 0;
            end
            OP_SRA: begin
                result = $signed(a) >>> b[$clog2(WIDTH)-1:0];
                carry = 0;
                overflow = 0;
            end
            OP_SLT: begin
                result = ($signed(a) < $signed(b)) ? 1 : 0;
                carry = 0;
                overflow = 0;
            end
            OP_SLTU: begin
                result = (a < b) ? 1 : 0;
                carry = 0;
                overflow = 0;
            end
            OP_PASS: begin
                result = a;
                carry = 0;
                overflow = 0;
            end
            default: begin
                result = {WIDTH{1'b0}};
                carry = 0;
                overflow = 0;
            end
        endcase

        zero = (result == 0);
        sign = result[WIDTH-1];
    end

endmodule
