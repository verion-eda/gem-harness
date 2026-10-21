// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use simulator::{SimulatorBuilder, signals::SignalPath};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    tracing::info!("=== Constant-D DFF Test ===");

    let mut dut = SimulatorBuilder::new(
        "examples/const-dff/build/gatelevel.gv".into(),
        "examples/const-dff/build/result.gemparts".into(),
        2,
    )
    .cpu()
    .build()?;
    dut.reset();

    let en: SignalPath = "en".parse()?;
    let normal_d: SignalPath = "normal_d".parse()?;
    let set_const: SignalPath = "set_const".parse()?;
    let reset_const: SignalPath = "reset_const".parse()?;
    let clk: SignalPath = "clk".parse()?;
    let q_const: SignalPath = "q_const".parse()?;
    let q_normal: SignalPath = "q_normal".parse()?;

    dut.poke_signal(&en, false)?;
    dut.poke_signal(&normal_d, false)?;
    dut.poke_signal(&set_const, true)?;
    dut.poke_signal(&reset_const, false)?;

    tracing::info!("Running clock cycles...");
    for i in 0..3 {
        dut.toggle(&clk)?; // rising edge
        dut.step();
        dut.toggle(&clk)?; // falling edge
        dut.step();
        dut.sync();

        let q_const_val = dut.peek_signal(&q_const)?.0;
        let q_normal_val = dut.peek_signal(&q_normal)?.0;
        tracing::info!(
            "Cycle {}: q_const0={}, q_normal={}",
            i,
            q_const_val,
            q_normal_val
        );

        if i >= 1 && q_const_val != 1 {
            tracing::error!("BUG: q_const0 should be 0 but is {}", q_const_val);
        }
    }

    let q_const_val = dut.peek_signal(&q_const)?.0;
    if q_const_val == 1 {
        tracing::info!("PASS: Constant-D DFFs working correctly");
    } else {
        tracing::error!("FAIL: q_const0={} (expected 1)", q_const_val);
    }

    Ok(())
}
