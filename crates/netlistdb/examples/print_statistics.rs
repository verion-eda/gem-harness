// SPDX-FileCopyrightText: Copyright (c) Zizheng Guo. Originally from eda-infra-rs (https://github.com/gzz2000/eda-infra-rs).
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
// SPDX-License-Identifier: Apache-2.0
use std::env;

use netlistdb::NetlistDB;

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::DEBUG)
        .init();
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() == 2 || args.len() == 3,
        "Usage: {} <verilog_path> [<top_module>]",
        args[0]
    );

    let db = NetlistDB::from_sverilog_file(
        &args[1],
        args.get(2).map(|x| x.as_ref()),
        &netlistdb::NoDirection,
    )
    .expect("Error parsing the verilog into netlist");

    println!("Benchmark statistics for {}", args[1]);
    println!("top module: {}", db.name);
    println!("num cells:  {}", db.num_cells);
    println!("num nets:   {}", db.num_nets);
    println!("num pins:   {}", db.num_pins);
}
