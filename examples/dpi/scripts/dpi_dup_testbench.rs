// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI duplicate registration testbench.
//
// Verifies that registering the same DPI symbol from multiple shared libraries
// is correctly rejected.
//
//   cargo run -r -p examples --bin dpi_dup_testbench

use simulator::SimulatorBuilder;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("=== Duplicate DPI Symbol Test ===");
    let ext = std::env::consts::DLL_EXTENSION;
    let dpi_lib = format!("examples/dpi/build/libdpi.{ext}");
    let dpi_lib_dup = format!("examples/dpi/build/libdpi_dup.{ext}");
    let result = SimulatorBuilder::new(
        "examples/dpi/build/gatelevel.gv".into(),
        "examples/dpi/build/result.gemparts".into(),
        2,
    )
    .cpu()
    .register_dpi("trace_event", |_args| vec![])
    .register_dpi_lib(dpi_lib)?
    .register_dpi_lib(dpi_lib_dup)?
    .build();

    let err_msg = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "build() should fail when the same DPI symbol is in multiple libraries"
        ),
    };
    assert!(
        err_msg.contains("multiple shared libraries"),
        "error should mention 'multiple shared libraries', got: {err_msg}"
    );
    tracing::info!("Duplicate symbol correctly rejected: {err_msg}");

    tracing::info!("TEST PASSED");
    Ok(())
}
