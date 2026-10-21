// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! CLI: gate-level Verilog → `.gemparts` GPU bitstream.
//!
//! This is the offline half of the toolchain. It parses synthesized
//! Verilog into an [`AIG`], runs [`staging`] and [`repcut`] partitioning,
//! then drives [`flatten`] to produce a serialized
//! [`FlattenedScriptV1`][flatten::FlattenedScriptV1] on disk.
//! The resulting `.gemparts` file is consumed by [`simulator`]
//! (interactive) or [`gem-simulator`] (one-shot) at runtime.
//!
//! The implementation is iterative: partitions that don't fit on one PE
//! are re-cut with a higher partition count, rather than committing to a
//! fixed partition count up front.
//!
//! [`simulator`]: ../simulator/
//! [`gem-simulator`]: ../gem_simulator/

use std::path::PathBuf;

use aig::{AIG, aigpdk::AIGPDKLeafPins};
use flatten::pe::{Partition, process_partitions};
use netlistdb::NetlistDB;
use rayon::prelude::*;
use repcut::RCHyperGraph;
use staging::build_staged_aigs;

/// Call builtin partitioner.
fn run_par(hg: &RCHyperGraph, num_parts: usize) -> Vec<Vec<usize>> {
    tracing::debug!("invoking partitioner (#parts {})", num_parts);
    // Handle the special case where num_parts = 1
    // mt-kahypar requires k >= 2, so we handle k=1 manually
    if num_parts == 1 {
        let mut parts = vec![vec![]; 1];
        // Put all vertices in the single partition
        for i in 0..hg.num_vertices() {
            parts[0].push(i);
        }
        return parts;
    }

    let parts_ids = hg.partition(num_parts);
    let mut parts = vec![vec![]; num_parts];
    for (i, part_id) in parts_ids.into_iter().enumerate() {
        parts[part_id].push(i);
    }
    parts
}

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
    /// The maximum allowance of layers for merging-induced degradations.
    ///
    /// By default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t = 0)]
    max_stage_degrad: usize,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .with_target(false)
        .init();

    let args = <SimulatorArgs as clap::Parser>::parse();
    tracing::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    )
    .expect("cannot build netlist");
    // Convert Netlist into Rust DAG representation. Fuse some elements in the
    // process and topologically sort.
    let aig = AIG::from_netlistdb(&netlistdb);
    println!(
        "netlist has {} pins, {} aig pins, {} and gates",
        netlistdb.num_pins,
        aig.num_aigpins,
        aig.and_gate_cache.len()
    );

    // Staged = Horizontal cutting of the AIG to decrease the logic depth
    let stageds = build_staged_aigs(&aig, &args.level_split);
    // Serialise as we need the stages in the simulator
    let mut stageds_out = args.parts_out.clone();
    stageds_out.set_extension("stageds");
    let f = std::fs::File::create(&stageds_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stageds).unwrap();

    let stages_effective_parts = stageds.iter().map(|&(l, r, ref staged)| {
        tracing::info!("interactive partitioning stage {}-{}", l, match r {
            usize::MAX => "max".to_string(),
            r => format!("{}", r)
        });

        let mut parts_indices_good = Vec::new();
        // always made sure that staged output pins are at fronts.
        let mut unrealized_endpoints = (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
        let mut division = 600;

        while !unrealized_endpoints.is_empty() {
            division = (division / 2).max(1);
            let num_parts = unrealized_endpoints.len().div_ceil(division);
            tracing::info!("current: {} endpoints, try {} parts", unrealized_endpoints.len(), num_parts);
            let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
            let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
            let mut parts_indices = run_par(&hg_ur, num_parts);
            for idcs in &mut parts_indices {
                for i in idcs {
                    *i = unrealized_endpoints[*i];
                }
            }
            let parts_try = parts_indices.par_iter()
                .map(|endpts| Partition::build_one(&aig, staged, endpts))
                .collect::<Vec<_>>();
            let mut new_unrealized_endpoints = Vec::new();
            for (idx, part_opt) in parts_indices.into_iter().zip(parts_try.into_iter()) {
                match part_opt {
                    Some(_part) => {
                        parts_indices_good.push(idx);
                    }
                    None => {
                        if idx.len() == 1 {
                            panic!("A single endpoint still cannot map, you need to increase level cut granularity.");
                        }
                        for endpt_i in idx {
                            new_unrealized_endpoints.push(endpt_i);
                        }
                    }
                }
            }
            new_unrealized_endpoints.sort_unstable();
            unrealized_endpoints = new_unrealized_endpoints;
        }

        tracing::info!("interactive partition completed: {} in total. merging started.",
                      parts_indices_good.len());

        let effective_parts = process_partitions(
            &aig, staged, parts_indices_good, args.max_stage_degrad
        ).unwrap();
        tracing::info!("after merging: {} parts.", effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}
