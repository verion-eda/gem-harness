// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI constant-input testbench.
//
// Verifies that DPI functions with constant inputs (folded parameters)
// work correctly through the DpiInputBit::Constant resolution path.
//
//   # Build:
//   cd examples/dpi && make sim_ci
//
//   # Or directly:
//   cargo run -r -p examples --bin dpi_const_input_testbench

use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("=== DPI Constant Input Testbench ===");

    let mut dut = SimulatorBuilder::new(
        "examples/dpi/build_const_input/gatelevel.gv".into(),
        "examples/dpi/build_const_input/result.gemparts".into(),
        2,
    )
    .cpu()
    .register_dpi("const_add", |args| {
        let a = args[0];
        let b = args[1];
        vec![a.wrapping_add(b)]
    })
    .build()?;

    tracing::info!("Inputs:  {:?}", dut.list_inputs());
    tracing::info!("Outputs: {:?}", dut.list_outputs());

    dut.reset();

    let clk: SignalPath = "clk".parse()?;
    let result_sp: SignalPath = "result".parse()?;

    let mut passed = 0;
    let mut failed = 0;

    for cycle in 1..=5usize {
        // Rising edge
        dut.poke_signal(&clk, 0)?;
        dut.step();
        dut.poke_signal(&clk, 1)?;
        dut.step();

        let result = dut.peek_signal(&result_sp)?;
        let expected = 49u64; // 42 + 7

        if result == expected {
            tracing::info!("Cycle {}: result = {} OK", cycle, result);
            passed += 1;
        } else {
            tracing::error!(
                "Cycle {}: result = {} (expected {})",
                cycle,
                result,
                expected
            );
            failed += 1;
        }
    }

    tracing::info!("=== Results: {}/{} passed ===", passed, passed + failed);

    if failed == 0 {
        tracing::info!("ALL TESTS PASSED");
        Ok(())
    } else {
        anyhow::bail!("{} test(s) failed", failed);
    }
}
