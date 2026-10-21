// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use simulator::{SimulatorBuilder, signals::SignalPath};

#[repr(u64)]
#[derive(Debug, Clone, Copy)]
enum AluOp {
    Add = 0b0000,
    Sub = 0b0001,
    And = 0b0010,
    Or = 0b0011,
    Xor = 0b0100,
    Sll = 0b0110,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== ALU Testbench (Multi-bit API) ===");

    // Build simulator
    let mut dut = SimulatorBuilder::new(
        "examples/alu/build/gatelevel.gv".into(),
        "examples/alu/build/result.gemparts".into(),
        2,
    )
    .build()?;

    tracing::info!("Simulator ready:");
    tracing::info!("  Inputs: {} signals", dut.num_inputs());
    tracing::info!("  Outputs: {} signals", dut.num_outputs());

    // Reset
    dut.reset();

    let test_cases = [
        // (a, b, op, expected_result, expected_zero)
        (10, 20, AluOp::Add, 30, false),
        (50, 20, AluOp::Sub, 30, false),
        (0xFF, 0x0F, AluOp::And, 0x0F, false),
        (0xF0, 0x0F, AluOp::Or, 0xFF, false),
        (0xAA, 0x55, AluOp::Xor, 0xFF, false),
        (1, 4, AluOp::Sll, 16, false),
        (10, 10, AluOp::Sub, 0, true),
    ];

    let a_sp: SignalPath = "a".parse()?;
    let b_sp: SignalPath = "b".parse()?;
    let op_sp: SignalPath = "op".parse()?;
    let result_sp: SignalPath = "result".parse()?;
    let zero_sp: SignalPath = "zero".parse()?;

    tracing::info!("Running ALU tests...");
    let mut passed = 0;

    for (test_num, (a, b, op, expected_res, expected_zero)) in
        test_cases.iter().enumerate()
    {
        // Set inputs using multi-bit poke_signal
        dut.poke_signal(&a_sp, *a)?;
        dut.poke_signal(&b_sp, *b)?;
        dut.poke_signal(&op_sp, *op as u64)?;

        // Clock the design
        dut.step();
        dut.sync();

        // Read outputs using multi-bit peek_signal
        let actual_res = dut.peek_signal(&result_sp)?;
        let actual_zero = dut.peek_signal(&zero_sp)? != 0;

        // Check results
        let test_passed =
            actual_res == *expected_res && actual_zero == *expected_zero;

        if test_passed {
            tracing::info!(
                "Test {}: 0x{:x} op({:?}) 0x{:x} = 0x{:x} (zero={}) ✓",
                test_num,
                a,
                op,
                b,
                actual_res,
                actual_zero
            );
            passed += 1;
        } else {
            tracing::error!(
                "Test {}: 0x{:x} op({:?}) 0x{:x} FAILED! Expected res=0x{:x}, zero={}, got res=0x{:x}, zero={}",
                test_num,
                a,
                op,
                b,
                expected_res,
                expected_zero,
                actual_res,
                actual_zero
            );
        }
    }

    tracing::info!("=== Test Results ===");
    tracing::info!("Passed: {}/{}", passed, test_cases.len());

    if passed == test_cases.len() {
        tracing::info!("🎉 ALL ALU TESTS PASSED! 🎉");
    } else {
        tracing::error!("❌ SOME ALU TESTS FAILED ❌");
        std::process::exit(1);
    }

    Ok(())
}
