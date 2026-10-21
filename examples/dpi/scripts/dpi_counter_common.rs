// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// Shared test logic for DPI counter testbenches.
//
// Both the Rust-closure and C-shared-library variants (CPU and GPU) use the
// same verification loop.  This module provides the common `run_counter_test`
// and `run_counter_test_with_trace` helpers so each testbench only needs to
// set up the `SimulatorBuilder` and call one function.

use simulator::{InteractiveSimulator, signals::SignalPath};

/// Run the saturating-counter verification loop.
///
/// Drives `inc = 1` for 10 rising edges and checks that `count` increments
/// correctly each cycle.  Returns `Ok(())` when all checks pass.
#[allow(dead_code)]
pub fn run_counter_test(dut: &mut InteractiveSimulator) -> anyhow::Result<()> {
    tracing::info!("Inputs:  {:?}", dut.list_inputs());
    tracing::info!("Outputs: {:?}", dut.list_outputs());

    dut.reset();

    let inc_sp: SignalPath = "inc".parse()?;
    let clk_sp: SignalPath = "clk".parse()?;
    let count_sp: SignalPath = "count".parse()?;

    let inc: u32 = 1;
    let mut passed = 0;
    let mut failed = 0;

    for cycle in 1..=10usize {
        dut.poke_signal(&inc_sp, inc as u64)?;

        // Rising edge: clk 0 -> 1
        dut.poke_signal(&clk_sp, 0)?;
        dut.step();
        dut.poke_signal(&clk_sp, 1)?;
        dut.step();

        let count = dut.peek_signal(&count_sp)?;
        let expected = cycle as u64;

        if count == expected {
            tracing::info!("Cycle {:2}: count = {} OK", cycle, count);
            passed += 1;
        } else {
            tracing::error!(
                "Cycle {:2}: count = {} (expected {})",
                cycle,
                count,
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

/// Run the saturating-counter verification loop *with* void-DPI trace
/// verification.
///
/// Same as [`run_counter_test`] but also checks that `trace_event` was called
/// with the expected count values (once per step, twice per cycle).
#[allow(dead_code)]
pub fn run_counter_test_with_trace(
    dut: &mut InteractiveSimulator,
    trace_log: &std::sync::Mutex<Vec<u32>>,
) -> anyhow::Result<()> {
    tracing::info!("Inputs:  {:?}", dut.list_inputs());
    tracing::info!("Outputs: {:?}", dut.list_outputs());

    dut.reset();

    let inc_sp: SignalPath = "inc".parse()?;
    let clk_sp: SignalPath = "clk".parse()?;
    let count_sp: SignalPath = "count".parse()?;

    let inc: u32 = 1;
    let mut passed = 0;
    let mut failed = 0;

    for cycle in 1..=10usize {
        dut.poke_signal(&inc_sp, inc as u64)?;

        // Rising edge: clk 0 -> 1
        dut.poke_signal(&clk_sp, 0)?;
        dut.step();
        dut.poke_signal(&clk_sp, 1)?;
        dut.step();

        let count = dut.peek_signal(&count_sp)?;
        let expected = cycle as u64;

        if count == expected {
            tracing::info!("Cycle {:2}: count = {} OK", cycle, count);
            passed += 1;
        } else {
            tracing::error!(
                "Cycle {:2}: count = {} (expected {})",
                cycle,
                count,
                expected
            );
            failed += 1;
        }
    }

    tracing::info!("=== Results: {}/{} passed ===", passed, passed + failed);

    // Verify trace_event (void DPI) was called with the expected count values.
    let events = trace_log.lock().unwrap();
    tracing::info!("trace_event calls: {:?}", *events);

    // trace_event runs on every step() call (2 per cycle: clk=0 then clk=1),
    // so each count value appears twice.
    let expected_events: Vec<u32> = (0..10).flat_map(|i| [i, i]).collect();
    if *events == expected_events {
        tracing::info!("trace_event: all {} calls correct", events.len());
    } else {
        failed += 1;
        tracing::error!(
            "trace_event mismatch: got {:?}, expected {:?}",
            *events,
            expected_events
        );
    }

    if failed == 0 {
        tracing::info!("ALL TESTS PASSED");
        Ok(())
    } else {
        anyhow::bail!("{} test(s) failed", failed);
    }
}
