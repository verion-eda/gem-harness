// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Example showing the new named signal access API
// Compare this to adder_testbench.rs which uses magic pin numbers
use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== Adder Testbench (Named Signals) ===");

    // Build simulator
    let mut dut = SimulatorBuilder::new(
        "examples/adder/build/gatelevel.gv".into(),
        "examples/adder/build/result.gemparts".into(),
        2,
    )
    .build()
    .expect("Failed to build simulator");

    tracing::info!(
        "Simulator ready with {} inputs and {} outputs",
        dut.num_inputs(),
        dut.num_outputs()
    );

    // Debug: print available inputs and outputs
    tracing::debug!("Available input AIG pins: {:?}", dut.list_inputs());
    tracing::debug!("Available output AIG pins: {:?}", dut.list_outputs());

    // Reset
    dut.reset();

    // Test all 4 combinations for 1-bit adder
    // NOW WITH NAMED SIGNALS - No magic numbers!
    let test_cases = [
        (false, false, false, false), // 0 + 0 = 0, carry=0
        (false, true, true, false),   // 0 + 1 = 1, carry=0
        (true, false, true, false),   // 1 + 0 = 1, carry=0
        (true, true, false, true),    // 1 + 1 = 0, carry=1
    ];

    let a_sp: SignalPath = "a".parse()?;
    let b_sp: SignalPath = "b".parse()?;
    let sum_sp: SignalPath = "sum".parse()?;
    let carry_sp: SignalPath = "carry".parse()?;

    tracing::info!("Running exhaustive tests...");
    let mut passed = 0;
    let mut failed = 0;

    for (test_num, (a, b, expected_sum, expected_carry)) in
        test_cases.iter().enumerate()
    {
        // Set inputs using signal NAMES instead of magic pin numbers
        dut.poke_signal(&a_sp, *a)?;
        dut.poke_signal(&b_sp, *b)?;

        // Clock the design
        dut.step();
        dut.sync();

        // Read outputs using signal NAMES
        let actual_sum = dut.peek_signal(&sum_sp)?;
        let actual_carry = dut.peek_signal(&carry_sp)?;

        // Check results
        let test_passed =
            actual_sum == *expected_sum && actual_carry == *expected_carry;

        if test_passed {
            tracing::info!(
                "Test {}: {} + {} = {} (carry={}) ✓",
                test_num,
                *a as u8,
                *b as u8,
                *actual_sum as u8,
                *actual_carry as u8
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
                *actual_sum as u8,
                *actual_carry as u8
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

    Ok(())
}
