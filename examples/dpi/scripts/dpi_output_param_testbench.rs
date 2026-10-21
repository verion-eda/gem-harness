// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
// DPI output-parameter testbench.
//
// Verifies that a DPI-C void function with an output parameter
// (`get_value(input int addr, output int data)`) works correctly
// through the simulator's DPI callback mechanism.
//
// The `get_value` callback receives the address as input and returns
// the data via the output parameter (packed as an output word).

use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    tracing::info!("=== DPI Output Parameter Testbench ===");

    let mut dut = SimulatorBuilder::new(
        "examples/dpi/build_output_param/gatelevel.gv".into(),
        "examples/dpi/build_output_param/result.gemparts".into(),
        2,
    )
    .cpu()
    // get_value(input int addr, output int data) -> void
    // args: [addr: 1 word (32-bit input)]
    // return: [data: 1 word (32-bit output parameter)]
    //
    // The output parameter is packed into the return vector just like
    // a return value would be — OUT_0 carries the output arg.
    .register_dpi("get_value", |args| {
        let addr = args[0];
        // Simple lookup: data = addr * 7 + 3
        let data = addr.wrapping_mul(7).wrapping_add(3);
        vec![data]
    })
    .build()?;

    tracing::info!("Inputs:  {:?}", dut.list_inputs());
    tracing::info!("Outputs: {:?}", dut.list_outputs());

    dut.reset();

    let addr_sp: SignalPath = "addr".parse()?;
    let clk: SignalPath = "clk".parse()?;
    let data_sp: SignalPath = "data".parse()?;

    let mut passed = 0;
    let mut failed = 0;

    let test_addrs: Vec<u32> = vec![0, 1, 2, 42, 0x100, 0xDEAD, 0xFFFF_FFFF];

    for (i, &addr) in test_addrs.iter().enumerate() {
        dut.poke_signal(&addr_sp, addr as u64)?;

        // Rising edge
        dut.poke_signal(&clk, 0)?;
        dut.step();
        dut.poke_signal(&clk, 1)?;
        dut.step();

        let data = dut.peek_signal(&data_sp)?;

        let expected = (addr.wrapping_mul(7).wrapping_add(3)) as u64;

        if data == expected {
            tracing::info!(
                "Test {}: addr=0x{:08X} => data=0x{:08X} ok",
                i,
                addr,
                data
            );
            passed += 1;
        } else {
            tracing::error!(
                "Test {}: addr=0x{:08X} => data=0x{:08X} (expected 0x{:08X})",
                i,
                addr,
                data,
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
