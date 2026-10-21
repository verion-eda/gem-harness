// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <vector>
#include <array>
#include <cstdint>
#include <iostream>
#include "Valu.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

// Enum class for ALU opcodes matching SystemVerilog definitions
enum class AluOp : uint8_t {
    OP_ADD  = 0b0000,  // 0
    OP_SUB  = 0b0001,  // 1
    OP_AND  = 0b0010,  // 2
    OP_OR   = 0b0011,  // 3
    OP_XOR  = 0b0100,  // 4
    OP_NOR  = 0b0101,  // 5
    OP_SLL  = 0b0110,  // 6 - shift left logical
    OP_SRL  = 0b0111,  // 7 - shift right logical
    OP_SRA  = 0b1000,  // 8 - shift right arithmetic
    OP_SLT  = 0b1001,  // 9 - set less than (signed)
    OP_SLTU = 0b1010,  // 10 - set less than (unsigned)
    OP_PASS = 0b1011   // 11 - pass-through a
};

// Test case structure to hold input and expected output
struct TestCase {
    AluOp op;
    uint32_t a;
    uint32_t b;
};

int main(int argc, char **argv) {
    // Initialize Verilators variables
    Verilated::commandArgs(argc, argv);

    // Create an instance of our module under test
    Valu *tb = new Valu;

    // Create VCD trace object
    Verilated::traceEverOn(true);
    VerilatedVcdC *trace = new VerilatedVcdC;
    
    // Enable tracing
    tb->trace(trace, 99);
    
    // Open VCD file for writing
    trace->open("input.vcd");

    std::vector<TestCase> test_vector = {
        {AluOp::OP_ADD, 5, 3},
        {AluOp::OP_SUB, 10, 4},
        {AluOp::OP_AND, 0b1100, 0b1010},
        {AluOp::OP_OR, 0b1100, 0b1010}
    };
    
    vluint64_t sim_time = 0;
    
    for (size_t i = 0; i < test_vector.size(); ++i) {
        // Set inputs
        tb->op = static_cast<uint8_t>(test_vector[i].op);
        tb->a = test_vector[i].a;
        tb->b = test_vector[i].b;
        
        // Evaluate the module
        tb->eval();
        
        // Dump trace at current time
        trace->dump(sim_time);
        sim_time += 1;
    }
    
    // Final dump
    trace->dump(sim_time);
    
    // Close trace file
    trace->close();
    trace=NULL;
    
    delete trace;
    delete tb;
    return 0;
}