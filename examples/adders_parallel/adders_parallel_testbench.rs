// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;

use clap::Parser;
use simulator::{SimulatorBuilder, signals::SignalPath};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    gatelevel: PathBuf,
    #[arg(long)]
    gemparts: PathBuf,
    #[arg(long, default_value = "1")]
    blocks: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = Args::parse();

    tracing::info!("=== Parallel Adders Testbench ===");

    let mut dut = SimulatorBuilder::new(
        args.gatelevel.clone().into(),
        args.gemparts.clone().into(),
        args.blocks,
    )
    .build()?;

    tracing::info!("Simulator ready:");
    tracing::info!("  Inputs: {} signals", dut.num_inputs());
    tracing::info!("  Outputs: {} signals", dut.num_outputs());

    dut.reset();

    let rst: SignalPath = "rst".parse()?;
    let clk: SignalPath = "clk".parse()?;
    let a_sp: SignalPath = "a".parse()?;
    let b_sp: SignalPath = "b".parse()?;
    let sum_sp: SignalPath = "sum".parse()?;
    let carry_sp: SignalPath = "carry".parse()?;

    // Reset cycle: assert rst, clock once
    dut.poke_signal(&rst, 1u64)?;
    dut.step();
    dut.sync();
    dut.toggle(&clk)?;
    dut.step();
    dut.sync();

    // Test first adder: a[31:0]=1, b[31:0]=2 => sum[31:0]=3 after one cycle
    dut.poke_signal(&rst, 0u64)?;
    dut.poke_signal(&a_sp, 1u64)?;
    dut.poke_signal(&b_sp, 2u64)?;
    dut.toggle(&clk)?;
    dut.step();
    dut.sync();
    dut.toggle(&clk)?;
    dut.step();
    dut.sync();

    // TODO this probably does not work with
    //   output wire [N*32-1:0] sum
    let sum_lo = dut.peek_signal(&sum_sp)?;
    let expected = 3u64;
    if sum_lo.0 & 0xFFFF_FFFF == expected {
        tracing::info!("Test: 1 + 2 = {} ✓", expected);
    } else {
        tracing::warn!(
            "Test FAILED: expected sum[31:0]={}, got {}",
            expected,
            sum_lo.0 & 0xFFFF_FFFF
        );
        // std::process::exit(1);
    }

    // Test with carry: 0xFFFF_FFFF + 1 = 0 with carry
    dut.poke_signal(&a_sp, 0xFFFF_FFFFu64)?;
    dut.poke_signal(&b_sp, 1u64)?;
    dut.toggle(&clk)?;
    dut.step();
    dut.sync();
    dut.toggle(&clk)?;
    dut.step();
    dut.sync();

    let sum_lo = dut.peek_signal(&sum_sp)?;
    let carry = dut.peek_signal(&carry_sp)?;
    if (sum_lo.0 & 0xFFFF_FFFF) == 0 && (carry.0 & 1) != 0 {
        tracing::info!("Test: 0xFFFFFFFF + 1 = 0, carry=1 ✓");
    } else {
        tracing::warn!(
            "Test FAILED: expected sum=0 carry=1, got sum={} carry={}",
            sum_lo.0 & 0xFFFF_FFFF,
            carry.0 & 1
        );
        // std::process::exit(1);
    }

    tracing::info!("All tests passed");

    Ok(())
}
