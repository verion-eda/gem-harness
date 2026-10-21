// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
module chained_adders #(
    parameter WIDTH = 8
)(
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire [WIDTH-1:0] c,
    input  wire [WIDTH-1:0] d,
    output wire [WIDTH:0]   sum_ab,
    output wire [WIDTH:0]   sum_cd,
    output wire [WIDTH+1:0] sum_total,
    output wire [WIDTH+1:0] sum_complex
);

    // Stage 1: Parallel adders (should create first vertical stage)
    assign sum_ab = a + b;
    assign sum_cd = c + d;
    
    // Stage 2: Sequential addition (should create second vertical stage)
    // This combines the results from stage 1
    assign sum_total = sum_ab + sum_cd;
    
    // Stage 3: More complex chain (should create third vertical stage)
    // This creates additional depth for testing multi-stage compilation
    wire [WIDTH+1:0] intermediate;
    assign intermediate = sum_total + a;
    assign sum_complex = intermediate + d;

endmodule
