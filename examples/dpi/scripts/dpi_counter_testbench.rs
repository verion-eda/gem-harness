// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI counter testbench (CPU).
//
// Verifies the saturating accumulator with DPI callbacks on the CPU backend.
// Tests both DpiMode::EveryCycle and DpiMode::Once, and writes a VCD trace.
//
//   cargo run -r -p examples --bin dpi_counter_testbench
//   cargo run -r -p examples --bin dpi_counter_testbench -- --dpi-lib

#[path = "dpi_counter_common.rs"]
mod dpi_counter_common;

use std::{
    fs::File,
    io::BufWriter,
    sync::{Arc, Mutex},
};

use clap::Parser;
use simulator::{DpiMode, SimulatorBuilder, signals::SignalPath};

#[derive(Parser)]
struct Args {
    /// Load sat_add from the C shared library instead of a Rust closure.
    #[arg(long)]
    dpi_lib: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    let mut builder = SimulatorBuilder::new(
        "examples/dpi/build/gatelevel.gv".into(),
        "examples/dpi/build/result.gemparts".into(),
        2,
    )
    .cpu();

    if args.dpi_lib {
        tracing::info!("=== DPI Counter Testbench (CPU, C shared library) ===");
        let ext = std::env::consts::DLL_EXTENSION;
        builder = builder
            .register_dpi_lib(format!("examples/dpi/build/libdpi.{ext}"))?
            .register_dpi("trace_event", |_| vec![]);
        let mut dut = builder.build()?;
        dpi_counter_common::run_counter_test(&mut dut)
    } else {
        tracing::info!("=== DPI Counter Testbench (CPU, Rust closures) ===");
        let trace_log: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let mut dut = builder
            .register_dpi("sat_add", |args| {
                let a = args[0] as i32;
                let b = args[1] as i32;
                vec![a.saturating_add(b) as u32]
            })
            .register_dpi("trace_event", {
                let trace_log = Arc::clone(&trace_log);
                move |args| {
                    trace_log.lock().unwrap().push(args[0]);
                    vec![]
                }
            })
            .build()?;

        // Enable VCD tracing for the count signal (32-bit bus → individual
        // bits).
        let count_handles: Vec<_> = (0..32)
            .filter_map(|i| dut.wire(&format!("count[{i}]")))
            .collect();
        assert!(!count_handles.is_empty(), "no 'count[N]' wires found");
        dut.trace(&count_handles);

        dpi_counter_common::run_counter_test_with_trace(&mut dut, &trace_log)?;

        // Write VCD trace.
        let vcd_path = "examples/dpi/build/dump.vcd";
        dut.write_vcd(BufWriter::new(File::create(vcd_path)?), "dpi_counter")?;
        tracing::info!("VCD trace written to {vcd_path}");

        // --- Test DpiMode::Once (runs once per reset) ---
        tracing::info!("=== DpiMode::Once Test ===");
        let once_log: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let mut dut = SimulatorBuilder::new(
            "examples/dpi/build/gatelevel.gv".into(),
            "examples/dpi/build/result.gemparts".into(),
            2,
        )
        .cpu()
        .register_dpi("sat_add", |args| {
            let a = args[0] as i32;
            let b = args[1] as i32;
            vec![a.saturating_add(b) as u32]
        })
        .register_dpi_with_mode("trace_event", DpiMode::Once, {
            let once_log = Arc::clone(&once_log);
            move |args| {
                once_log.lock().unwrap().push(args[0]);
                vec![]
            }
        })
        .build()?;

        // First reset: Once callback should fire.
        dut.reset();
        let after_first_reset = once_log.lock().unwrap().len();
        tracing::info!(
            "After first reset: {} Once-callback calls",
            after_first_reset,
        );

        // Step a few cycles: Once callback should NOT fire.
        let clk: SignalPath = "clk".parse()?;
        for _ in 0..5 {
            dut.poke_signal(&clk, 0)?;
            dut.step();
            dut.poke_signal(&clk, 1)?;
            dut.step();
        }
        let after_steps = once_log.lock().unwrap().len();
        tracing::info!(
            "After 5 cycles: {} Once-callback calls (should be same as after reset)",
            after_steps,
        );

        // Second reset: Once callback should fire again.
        dut.reset();
        let after_second_reset = once_log.lock().unwrap().len();
        tracing::info!(
            "After second reset: {} Once-callback calls",
            after_second_reset,
        );

        // Verify: called during each reset, not during steps.
        assert!(
            after_first_reset > 0,
            "Once callback should fire during first reset"
        );
        assert_eq!(
            after_steps, after_first_reset,
            "Once callback should not fire during step()"
        );
        assert!(
            after_second_reset > after_steps,
            "Once callback should fire again during second reset"
        );
        tracing::info!("DpiMode::Once: all assertions passed");

        tracing::info!("ALL TESTS PASSED");
        Ok(())
    }
}
