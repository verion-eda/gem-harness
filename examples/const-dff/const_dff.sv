// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
module const_dff(
    input wire clk,
    input wire set_const,    // set r_const to 1
    input wire reset_const,  // reset r_const to 0
    input wire en,           // enable for normal DFF
    input wire normal_d,     // normal data input
    output wire q_const,     // output from constant-D DFF (set/reset)
    output wire q_normal     // output from normal DFF
);
    reg r_const;
    reg r_normal;

    // DFF that captures constant 1 when set, constant 0 when reset
    // This creates TWO activations with constant data (d_iv=0 and d_iv=1)
    always @(posedge clk) begin
        if (set_const)
            r_const <= 1'b1;   // constant D=1
        else if (reset_const)
            r_const <= 1'b0;   // constant D=0
    end

    // Normal DFF for comparison
    always @(posedge clk) begin
        if (en)
            r_normal <= normal_d;
    end

    assign q_const = r_const;
    assign q_normal = r_normal;
endmodule
