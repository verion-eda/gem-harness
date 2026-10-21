// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! DPI pipe throughput benchmark
//!
//! Measures cycles/second for single-step vs batched_step on a minimal design
//! with two EveryCycle DPI callbacks (sat_add, trace_event).

use std::time::Instant;

use simulator::{SimulatorBuilder, ToggleClock, signals::SignalPath};

const GATELEVEL: &str = "examples/dpi/build/gatelevel.gv";
const GEMPARTS: &str = "examples/dpi/build/result.gemparts";

const WARMUP_CYCLES: usize = 1_000;
const BENCH_CYCLES: usize = 10_000;
const NUM_BLOCKS: usize = 1;

fn build_sim() -> anyhow::Result<simulator::InteractiveSimulator> {
    SimulatorBuilder::new(GATELEVEL.into(), GEMPARTS.into(), NUM_BLOCKS)
        .register_dpi("sat_add", |args| {
            let result = (args[0] as i32).saturating_add(args[1] as i32);
            vec![result as u32]
        })
        .register_dpi("trace_event", |_args| vec![])
        .build()
}

fn bench_single_step(
    sim: &mut simulator::InteractiveSimulator,
    cycles: usize,
) -> anyhow::Result<std::time::Duration> {
    let inc: SignalPath = "inc".parse()?;
    let clk: SignalPath = "clk".parse()?;

    sim.poke_signal(&inc, 1)?;

    let start = Instant::now();
    for _ in 0..cycles {
        sim.poke_signal(&clk, 0)?;
        sim.step();
        sim.poke_signal(&clk, 1)?;
        sim.step();
    }
    Ok(start.elapsed())
}

fn bench_batched(
    sim: &mut simulator::InteractiveSimulator,
    cycles: usize,
    half_cycle: &mut u64,
) -> anyhow::Result<std::time::Duration> {
    let inc: SignalPath = "inc".parse()?;
    let clk_val = if *half_cycle % 2 == 0 { 0u64 } else { 1 };
    let clk: SignalPath = "clk".parse()?;

    sim.poke_signal(&inc, 1)?;
    sim.poke_signal(&clk, clk_val)?;

    let start = Instant::now();
    let half_cycles_retired = sim.batched_step(
        cycles * 2,
        &[ToggleClock {
            signal: "clk",
            freq: 1,
            phase: 1,
        }],
        None,
    );
    let elapsed = start.elapsed();
    *half_cycle += half_cycles_retired as u64;
    Ok(elapsed)
}

fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .try_init();

    tracing::info!(
        "DPI pipe benchmark — {} warm-up + {} measured cycles",
        WARMUP_CYCLES,
        BENCH_CYCLES
    );
    tracing::info!("Design: {GATELEVEL}");

    // --- Single-step ---
    {
        let mut sim = build_sim()?;
        sim.reset();

        bench_single_step(&mut sim, WARMUP_CYCLES)?;

        let elapsed = bench_single_step(&mut sim, BENCH_CYCLES)?;
        let cycles_per_sec = BENCH_CYCLES as f64 / elapsed.as_secs_f64();
        tracing::info!(
            "single-step : {:>10.3} ms  →  {:>10.0} cycles/s  ({:.3} µs/cycle)",
            elapsed.as_secs_f64() * 1000.0,
            cycles_per_sec,
            elapsed.as_secs_f64() * 1e6 / BENCH_CYCLES as f64,
        );
    }

    // --- Batched step ---
    {
        let mut sim = build_sim()?;
        sim.reset();

        let mut half_cycle = 0u64;

        bench_batched(&mut sim, WARMUP_CYCLES, &mut half_cycle)?;

        let elapsed = bench_batched(&mut sim, BENCH_CYCLES, &mut half_cycle)?;
        let cycles_per_sec = BENCH_CYCLES as f64 / elapsed.as_secs_f64();
        tracing::info!(
            "batched-step: {:>10.3} ms  →  {:>10.0} cycles/s  ({:.3} µs/cycle)",
            elapsed.as_secs_f64() * 1000.0,
            cycles_per_sec,
            elapsed.as_secs_f64() * 1e6 / BENCH_CYCLES as f64,
        );
    }

    Ok(())
}
