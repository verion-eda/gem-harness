// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Module: timer
// Description: A simple down-counter that asserts 'done' when it reaches 0.
// ============================================================================
module timer #(
    parameter WIDTH = 32
  )(
    input  wire             clk,
    input  wire             rst,
    input  wire             load,
    input  wire [WIDTH-1:0] load_value,
    output wire             done
  );
  reg [WIDTH-1:0] count;

  always @(posedge clk)
  begin
    if (rst)
    begin
      count <= load_value;
    end
    else if (load)
    begin
      count <= load_value;
    end
    else if (count > 0)
    begin
      count <= count - 1;
    end
  end

  assign done = (count == 0);

endmodule

// ============================================================================
// Module: controller
// Description: FSM for traffic light sequence.
//              Red -> Green -> Yellow -> Red ...
// ============================================================================
module controller (
    input  wire        clk,
    input  wire        rst,
    input  wire        timer_done,
    output reg         timer_load,
    output reg  [31:0] timer_value,
    output reg         red,
    output reg         yellow,
    output reg         green
  );
  // State encoding
  localparam S_RED    = 2'b00;
  localparam S_GREEN  = 2'b01;
  localparam S_YELLOW = 2'b10;

  // Durations (in clock cycles) - simplified for simulation
  localparam TIME_RED    = 32'd50;
  localparam TIME_GREEN  = 32'd40;
  localparam TIME_YELLOW = 32'd10;

  reg [1:0] state, next_state;

  // State register
  always @(posedge clk)
  begin
    if (rst)
      state <= S_RED;
    else
      state <= next_state;
  end

  // Next state and output logic
  always @*
  begin
    // Defaults
    next_state  = state;
    timer_load  = 0;
    timer_value = 0;
    red         = 0;
    yellow      = 0;
    green       = 0;

    case (state)
      S_RED:
      begin
        red = 1;
        if (rst)
        begin
          timer_load  = 1;
          timer_value = TIME_RED;
        end
        else if (timer_done)
        begin
          next_state  = S_GREEN;
          timer_load  = 1;
          timer_value = TIME_GREEN;
        end
      end
      S_GREEN:
      begin
        green = 1;
        if (timer_done)
        begin
          next_state  = S_YELLOW;
          timer_load  = 1;
          timer_value = TIME_YELLOW;
        end
      end
      S_YELLOW:
      begin
        yellow = 1;
        if (timer_done)
        begin
          next_state  = S_RED;
          timer_load  = 1;
          timer_value = TIME_RED;
        end
      end
      default:
        next_state = S_RED;
    endcase
  end

endmodule

// ============================================================================
// Module: traffic_light_top
// Description: Top-level module connecting Timer and Controller.
// ============================================================================
module traffic_light_top (
    input  wire clk,
    input  wire rst,
    output wire red,
    output wire yellow,
    output wire green
  );

  (* keep *) wire        timer_done;
  (* keep *) wire        timer_load;
  (* keep *) wire [31:0] timer_value;

  // Instantiate Timer
  timer #(
          .WIDTH(32)
        ) u_timer (
          .clk       (clk),
          .rst       (rst),
          .load      (timer_load),
          .load_value(timer_value),
          .done      (timer_done)
        );

  // Instantiate Controller
  controller u_controller (
               .clk        (clk),
               .rst        (rst),
               .timer_done (timer_done),
               .timer_load (timer_load),
               .timer_value(timer_value),
               .red        (red),
               .yellow     (yellow),
               .green      (green)
             );

endmodule
