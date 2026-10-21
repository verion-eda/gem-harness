// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::{fs::File, io::BufWriter, path::PathBuf};

use simulator::SimulatorBuilder;

mod traffic_light_wrapper;
use traffic_light_wrapper::TrafficLightWrapper;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== Traffic Light Testbench ===");

    let example_dir = PathBuf::from("examples/traffic_light");
    let gatelevel_path = example_dir.join("build/gatelevel.gv");
    let gemparts_path = example_dir.join("build/result.gemparts");
    let vcd_path = example_dir.join("build/dump.vcd");

    // Build simulator wrapper
    let mut dut = TrafficLightWrapper::new(
        SimulatorBuilder::new(gatelevel_path, gemparts_path, 2)
            .top_module("traffic_light_top".into()),
    )?;

    tracing::info!("Simulator ready:");
    tracing::info!("  Inputs: {} signals", dut.sim.num_inputs());
    tracing::info!("  Outputs: {} signals", dut.sim.num_outputs());

    // Setup tracing
    // We want to trace the top level IO and internal signals if possible
    // Note: synthesized names might differ, but top level ports usually
    // preserved dut.trace(vec![
    //     "red".to_string(),
    //     "yellow".to_string(),
    //     "green".to_string(),
    //     // "clk".to_string(), // clk is handled specially usually, or
    // implicit     // "rst".to_string()
    // ]);

    // Trace all signals marked (* keep *) in the source (timer_done,
    // timer_load, timer_value[*]) so they appear in the VCD dump.
    let keep = dut.sim.keep_signals().to_vec();
    tracing::info!(
        "  Keep signals: {:?}",
        keep.iter()
            .map(|&h| dut.sim.wire_name(h))
            .collect::<Vec<_>>()
    );
    dut.sim.trace(&keep);

    // Reset
    dut.sim.reset();
    dut.rst = 1;
    dut.clk = 0;

    tracing::info!("Starting simulation...");

    // Initial state check (Red)
    dut.eval(); // Apply reset

    let mut passed = true;

    // Run for some cycles and monitor
    // Expected sequence: Red (50) -> Green (40) -> Yellow (10) -> Red
    // Note: Timings might be slightly off due to FSM transitions, so we check
    // windows or transitions

    for cycle in 0..150 {
        dut.rst = if cycle < 2 { 1 } else { 0 };

        // Falling edge
        dut.clk = 0;
        dut.eval();

        // Rising edge (state updates here)
        dut.clk = 1;
        dut.eval();

        let red = dut.red;
        let yellow = dut.yellow;
        let green = dut.green;

        // Simple logging of transitions
        if cycle == 1
            || cycle == 50
            || cycle == 51
            || cycle == 90
            || cycle == 91
            || cycle == 100
            || cycle == 101
        {
            tracing::info!(
                "Cycle {}: R={} Y={} G={}",
                cycle,
                red,
                yellow,
                green
            );
        }

        // Basic Assertions (shifted by 1 cycle: simulator uses state at start
        // of cycle, so transitions appear one cycle later)
        if cycle < 52 {
            // Should be RED
            if !red {
                tracing::error!(
                    "Cycle {}: Expected RED, got R={} Y={} G={}",
                    cycle,
                    red,
                    yellow,
                    green
                );
                passed = false;
            }
        } else if cycle >= 53 && cycle < 93 {
            // Should be GREEN
            if !green {
                tracing::error!(
                    "Cycle {}: Expected GREEN, got R={} Y={} G={}",
                    cycle,
                    red,
                    yellow,
                    green
                );
                passed = false;
            }
        } else if cycle >= 94 && cycle < 104 {
            // Should be YELLOW (transition visible one cycle after GREEN ends)
            if !yellow {
                tracing::error!(
                    "Cycle {}: Expected YELLOW, got R={} Y={} G={}",
                    cycle,
                    red,
                    yellow,
                    green
                );
                passed = false;
            }
        }
    }

    if passed {
        tracing::info!("🎉 ALL TESTS PASSED! 🎉");
    } else {
        tracing::error!("❌ SOME TESTS FAILED ❌");
    }

    dut.sim.write_vcd(
        BufWriter::new(File::create(&vcd_path)?),
        "traffic_light_top",
    )?;
    tracing::info!("VCD dump written to {}", vcd_path.display());
    Ok(())
}
