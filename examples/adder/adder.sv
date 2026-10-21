// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
module adder(input wire a, input wire b, output wire sum, output wire carry);
  assign {carry, sum} = a + b;
endmodule
