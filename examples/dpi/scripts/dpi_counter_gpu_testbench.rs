// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI counter GPU testbench.
//
// Verifies the saturating accumulator with DPI callbacks on the GPU backend.
//
//   cargo run -r -p examples --bin dpi_counter_gpu_testbench --features
// cuda   cargo run -r -p examples --bin dpi_counter_gpu_testbench
// --features cuda -- --dpi-lib

#[path = "dpi_counter_common.rs"]
mod dpi_counter_common;

use std::sync::{Arc, Mutex};

use clap::Parser;
use simulator::SimulatorBuilder;

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
    );

    if args.dpi_lib {
        tracing::info!("=== DPI Counter Testbench (GPU, C shared library) ===");
        let ext = std::env::consts::DLL_EXTENSION;
        builder = builder
            .register_dpi_lib(format!("examples/dpi/build/libdpi.{ext}"))?
            .register_dpi("trace_event", |_| vec![]);
        let mut dut = builder.build()?;
        dpi_counter_common::run_counter_test(&mut dut)
    } else {
        tracing::info!("=== DPI Counter Testbench (GPU, Rust closures) ===");
        let trace_log: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let trace_log_cb = Arc::clone(&trace_log);
        let mut dut = builder
            .register_dpi("sat_add", |args| {
                let a = args[0] as i32;
                let b = args[1] as i32;
                vec![a.saturating_add(b) as u32]
            })
            .register_dpi("trace_event", move |args| {
                trace_log_cb.lock().unwrap().push(args[0]);
                vec![]
            })
            .build()?;
        dpi_counter_common::run_counter_test_with_trace(&mut dut, &trace_log)
    }
}
