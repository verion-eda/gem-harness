// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// This is more or less how I imagine a interctive simulator.
// Check out Verilator scripts as inspiration (https://itsembedded.com/dhd/verilator_4/)

use std::{fs::File, io::BufWriter};

use simulator::{PinType, SimulatorBuilder};

const PIN_A: PinType = 1;
const PIN_B: PinType = 2;
const PIN_SUM: PinType = 10;
const PIN_CARRY: PinType = 6;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== Adder Testbench (Verilator-Style) ===");

    // Build simulator
    let mut dut = SimulatorBuilder::new(
        "examples/adder/build/gatelevel.gv".into(),
        "examples/adder/build/result.gemparts".into(),
        2,
    )
    .build()?;

    tracing::info!("Simulator ready:");
    tracing::info!("  Inputs: {:?}", dut.list_inputs());
    tracing::info!("  Outputs: {:?}", dut.list_outputs());

    // Reset
    dut.reset();

    // Test all 4 combinations for 1-bit adder
    let test_cases = [
        (false, false, false, false), // 0 + 0 = 0, carry=0
        (false, true, true, false),   // 0 + 1 = 1, carry=0
        (true, false, true, false),   // 1 + 0 = 1, carry=0
        (true, true, false, true),    // 1 + 1 = 0, carry=1
    ];

    tracing::info!("Running exhaustive tests...");
    let mut passed = 0;
    let mut failed = 0;

    let sum = dut.wire("sum").expect("signal 'sum' should exist");
    let carry = dut.wire("carry").expect("signal 'carry' should exist");
    dut.trace(&[sum, carry]);

    for (test_num, (a, b, expected_sum, expected_carry)) in
        test_cases.iter().enumerate()
    {
        // Set inputs
        dut.poke(PIN_A, *a);
        dut.poke(PIN_B, *b);

        // Clock the design
        dut.step();
        dut.sync();

        // Read outputs
        let actual_sum = dut.peek(PIN_SUM);
        let actual_carry = dut.peek(PIN_CARRY);

        // Check results
        let test_passed =
            actual_sum == *expected_sum && actual_carry == *expected_carry;

        if test_passed {
            tracing::info!(
                "Test {}: {} + {} = {} (carry={}) ✓",
                test_num,
                *a as u8,
                *b as u8,
                actual_sum as u8,
                actual_carry as u8
            );
            passed += 1;
        } else {
            tracing::error!(
                "Test {}: {} + {} FAILED! Expected sum={}, carry={}, got sum={}, carry={}",
                test_num,
                *a as u8,
                *b as u8,
                *expected_sum as u8,
                *expected_carry as u8,
                actual_sum as u8,
                actual_carry as u8
            );
            failed += 1;
        }
    }

    tracing::info!("=== Test Results ===");
    tracing::info!("Passed: {}/{}", passed, test_cases.len());
    tracing::info!("Failed: {}/{}", failed, test_cases.len());

    if failed == 0 {
        tracing::info!("🎉 ALL TESTS PASSED! 🎉");
    } else {
        tracing::error!("❌ SOME TESTS FAILED ❌");
        std::process::exit(1);
    }

    dut.write_vcd(
        BufWriter::new(File::create("examples/adder/build/dump.vcd.trace")?),
        "adder_top_module",
    )?;

    Ok(())
}
