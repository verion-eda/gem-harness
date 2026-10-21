// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== Chained Adders Testbench ===");

    // Build simulator
    let mut dut = SimulatorBuilder::new(
        "examples/chained-adders/build/gatelevel.gv".into(),
        "examples/chained-adders/build/result.gemparts".into(),
        2,
    )
    .build()?;

    tracing::info!("Simulator ready:");
    tracing::info!("  Inputs: {} signals", dut.num_inputs());
    tracing::info!("  Outputs: {} signals", dut.num_outputs());

    // Reset
    dut.reset();

    let test_cases = [
        // (a, b, c, d, expected_sum_ab, expected_sum_cd, expected_sum_total,
        // expected_sum_complex)
        (10, 20, 30, 40, 30, 70, 100, 150),
        (5, 5, 10, 10, 10, 20, 30, 45),
        (1, 2, 3, 4, 3, 7, 10, 15),
        (255, 1, 128, 64, 256, 192, 448, 767),
        (100, 50, 25, 25, 150, 50, 200, 325),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (7, 8, 9, 10, 15, 19, 34, 51),
    ];

    let a_sp: SignalPath = "a".parse()?;
    let b_sp: SignalPath = "b".parse()?;
    let c_sp: SignalPath = "c".parse()?;
    let d_sp: SignalPath = "d".parse()?;
    let sum_ab_sp: SignalPath = "sum_ab".parse()?;
    let sum_cd_sp: SignalPath = "sum_cd".parse()?;
    let sum_total_sp: SignalPath = "sum_total".parse()?;
    let sum_complex_sp: SignalPath = "sum_complex".parse()?;

    tracing::info!("Running chained adder tests...");
    let mut passed = 0;

    for (
        test_num,
        (
            a,
            b,
            c,
            d,
            expected_ab,
            expected_cd,
            expected_total,
            expected_complex,
        ),
    ) in test_cases.iter().enumerate()
    {
        // Set inputs
        dut.poke_signal(&a_sp, *a)?;
        dut.poke_signal(&b_sp, *b)?;
        dut.poke_signal(&c_sp, *c)?;
        dut.poke_signal(&d_sp, *d)?;

        // Clock the design
        dut.step();
        dut.sync();

        // Read outputs
        let actual_ab = dut.peek_signal(&sum_ab_sp)?;
        let actual_cd = dut.peek_signal(&sum_cd_sp)?;
        let actual_total = dut.peek_signal(&sum_total_sp)?;
        let actual_complex = dut.peek_signal(&sum_complex_sp)?;

        // Check results
        let test_passed = actual_ab == *expected_ab
            && actual_cd == *expected_cd
            && actual_total == *expected_total
            && actual_complex == *expected_complex;

        if test_passed {
            tracing::info!(
                "Test {}: a={}, b={}, c={}, d={} => sum_ab={}, sum_cd={}, sum_total={}, sum_complex={} ✓",
                test_num,
                a,
                b,
                c,
                d,
                actual_ab,
                actual_cd,
                actual_total,
                actual_complex
            );
            passed += 1;
        } else {
            tracing::error!(
                "Test {}: FAILED! a={}, b={}, c={}, d={}",
                test_num,
                a,
                b,
                c,
                d
            );
            tracing::error!(
                "  Expected: sum_ab={}, sum_cd={}, sum_total={}, sum_complex={}",
                expected_ab,
                expected_cd,
                expected_total,
                expected_complex
            );
            tracing::error!(
                "  Got:      sum_ab={}, sum_cd={}, sum_total={}, sum_complex={}",
                actual_ab,
                actual_cd,
                actual_total,
                actual_complex
            );
        }
    }

    tracing::info!("=== Test Results ===");
    tracing::info!("Passed: {}/{}", passed, test_cases.len());

    if passed == test_cases.len() {
        tracing::info!("🎉 ALL TESTS PASSED! 🎉");
    } else {
        tracing::error!("❌ SOME TESTS FAILED ❌");
        std::process::exit(1);
    }

    Ok(())
}
