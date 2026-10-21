// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Trace the complete signal chain from RTL port name to global memory
//!
//! This example demonstrates how to:
//! 1. Load a gate-level netlist and understand its structure
//! 2. Map from port names (like "a") to NetlistDB pin IDs
//! 3. Map from pin IDs to AIG pins
//! 4. Map from AIG pins to global memory offsets
//! 5. Actually poke/peek values in global memory

use aig::{AIG, DriverType, aigpdk::AIGPDKLeafPins};
use flatten::{FlattenedScriptV1, pe::Partition};
use staging::build_staged_aigs;
use indexmap::IndexMap;
use netlistdb::NetlistDB;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Complete Signal Chain: Port Name → Global Memory             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // STEP 1: Load the gate-level netlist into NetlistDB
    // =========================================================================
    println!("━━━ STEP 1: Load NetlistDB from gate-level Verilog ━━━\n");

    let netlistdb = NetlistDB::from_sverilog_file(
        "examples/alu/gatelevel.gv",
        None, // auto-detect top module
        &AIGPDKLeafPins(),
    )
    .expect("Failed to load netlist");

    println!("  Design: {}", netlistdb.name);
    println!("  Cells: {}", netlistdb.num_cells);
    println!("  Pins: {}", netlistdb.num_pins);
    println!("  Nets: {}", netlistdb.num_nets);
    println!();

    // Show all top-level ports and their pin IDs
    println!("  Top-level ports (portname2pinid):");
    for ((port_name, bit_idx), &pin_id) in &netlistdb.portname2pinid {
        let idx_str = match bit_idx {
            Some(i) => format!("[{}]", i),
            None => String::new(),
        };
        println!(
            "    Port: {}{} → NetlistDB pin_id: {}",
            port_name, idx_str, pin_id
        );
    }
    println!();

    // =========================================================================
    // STEP 2: Convert NetlistDB to AIG
    // =========================================================================
    println!("━━━ STEP 2: Convert to AIG (And-Inverter Graph) ━━━\n");

    let aig = AIG::from_netlistdb(&netlistdb);

    println!("  AIG pins: {}", aig.num_aigpins);
    println!("  Primary outputs: {:?}", aig.primary_outputs);
    println!();

    // Show the pin2aigpin_iv mapping
    println!("  NetlistDB pin → AIG pin mapping (pin2aigpin_iv):");
    for ((port_name, bit_idx), &pin_id) in &netlistdb.portname2pinid {
        let aigpin_iv = aig.pin2aigpin_iv[pin_id];
        let aigpin = aigpin_iv >> 1;
        let inverted = (aigpin_iv & 1) != 0;
        let idx_str = match bit_idx {
            Some(i) => format!("[{}]", i),
            None => String::new(),
        };
        println!(
            "    {}{}: pin_id {} → AIG pin {} {}",
            port_name,
            idx_str,
            pin_id,
            aigpin,
            if inverted { "(inverted)" } else { "" }
        );
    }
    println!();

    // Show all AIG drivers
    println!("  AIG driver types:");
    for (aigpin, driver) in aig.drivers.iter().enumerate() {
        let driver_str = match driver {
            DriverType::Tie0 => "Constant 0".to_string(),
            DriverType::InputPort(pid) => {
                // Find the port name for this pin_id
                let name = netlistdb
                    .portname2pinid
                    .iter()
                    .find(|(_, &id)| id == *pid)
                    .map(|((n, idx), _)| {
                        let idx_str = match idx {
                            Some(i) => format!("[{}]", i),
                            None => String::new(),
                        };
                        format!("{}{}", n, idx_str)
                    })
                    .unwrap_or_else(|| format!("pin_{}", pid));
                format!("InputPort({}) = \"{}\"", pid, name)
            }
            DriverType::AndGate(a, b) => {
                let inv_a = if a & 1 != 0 { "!" } else { "" };
                let inv_b = if b & 1 != 0 { "!" } else { "" };
                format!("AND({}pin{}, {}pin{})", inv_a, a >> 1, inv_b, b >> 1)
            }
            DriverType::DFF(idx) => format!("DFF({})", idx),
            DriverType::SRAM(idx) => format!("SRAM({})", idx),
            DriverType::InputClockFlag(pid, edge) => format!("ClockFlag({}, {})", pid, edge),
        };
        println!("    AIG pin {}: {}", aigpin, driver_str);
    }
    println!();

    // =========================================================================
    // STEP 3: Build the flattened script (bitstream compilation)
    // =========================================================================
    println!("━━━ STEP 3: Build Flattened Script (Bitstream) ━━━\n");

    // Build input layout (which AIG pins are inputs)
    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }
    println!(
        "  Input layout (AIG pins that are inputs): {:?}",
        input_layout
    );

    // Load pre-compiled partitions
    let f = std::fs::File::open("examples/adder/result.gemparts").expect("Failed to open gemparts");
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> =
        serde_bare::from_reader(&mut buf).expect("Failed to parse gemparts");

    // Build staged AIGs
    let stageds = build_staged_aigs(&aig, &[]); // empty level_split

    // Create the flattened script
    let num_blocks = 2;
    let script = FlattenedScriptV1::from(
        &aig,
        &stageds
            .iter()
            .map(|(_, _, staged)| staged)
            .collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        num_blocks,
        input_layout.clone(),
    );

    println!(
        "  State size: {} u32 words ({} bits)",
        script.reg_io_state_size,
        script.reg_io_state_size * 32
    );
    println!("  SRAM storage: {} u32 words", script.sram_storage_size);
    println!();

    // =========================================================================
    // STEP 4: Examine the input_map (AIG pin → global memory offset)
    // =========================================================================
    println!("━━━ STEP 4: Input Map (AIG pin → Global Memory Offset) ━━━\n");

    println!("  input_map contents:");
    for (&aigpin, &state_offset) in &script.input_map {
        let word_idx = state_offset / 32;
        let bit_idx = state_offset % 32;

        // Find the port name for this AIG pin
        let port_name = if let DriverType::InputPort(pid) = &aig.drivers[aigpin] {
            netlistdb
                .portname2pinid
                .iter()
                .find(|(_, &id)| id == *pid)
                .map(|((n, idx), _)| {
                    let idx_str = match idx {
                        Some(i) => format!("[{}]", i),
                        None => String::new(),
                    };
                    format!("{}{}", n, idx_str)
                })
                .unwrap_or_else(|| format!("pin_{}", pid))
        } else {
            "???".to_string()
        };

        println!(
            "    AIG pin {} (\"{}\") → state_offset {} (word[{}] bit {})",
            aigpin, port_name, state_offset, word_idx, bit_idx
        );
    }
    println!();

    // =========================================================================
    // STEP 5: Examine the output_map
    // =========================================================================
    println!("━━━ STEP 5: Output Map (AIG pin → Global Memory Offset) ━━━\n");

    println!("  output_map contents:");
    for (&aigpin, &state_offset_with_inv) in &script.output_map {
        let invert = (state_offset_with_inv & 1) != 0;
        let state_offset = state_offset_with_inv >> 1;
        let word_idx = state_offset / 32;
        let bit_idx = state_offset % 32;

        println!(
            "    AIG pin {} → state_offset {} (word[{}] bit {}) {}",
            aigpin,
            state_offset,
            word_idx,
            bit_idx,
            if invert { "[INVERTED]" } else { "" }
        );
    }
    println!();

    // =========================================================================
    // STEP 6: Complete lookup example - find port "a" in global memory
    // =========================================================================
    println!("━━━ STEP 6: Complete Chain for Port \"a\" ━━━\n");

    let port_name = "a";
    println!("  Goal: Set port \"{}\" = 1 in global memory\n", port_name);

    // Step 6a: Port name → NetlistDB pin_id
    let pin_id = netlistdb
        .portname2pinid
        .get(&(port_name.into(), None))
        .copied()
        .expect("Port not found in netlist");
    println!(
        "  6a. Port \"{}\" → NetlistDB pin_id = {}",
        port_name, pin_id
    );

    // Step 6b: NetlistDB pin_id → AIG pin
    let aigpin_iv = aig.pin2aigpin_iv[pin_id];
    let aigpin = aigpin_iv >> 1;
    let inverted = (aigpin_iv & 1) != 0;
    println!(
        "  6b. NetlistDB pin_id {} → AIG pin {} {}",
        pin_id,
        aigpin,
        if inverted { "(inverted)" } else { "" }
    );

    // Step 6c: AIG pin → Global memory offset
    let state_offset = script
        .input_map
        .get(&aigpin)
        .copied()
        .expect("AIG pin not in input_map");
    let word_idx = state_offset / 32;
    let bit_idx = state_offset % 32;
    println!(
        "  6c. AIG pin {} → state_offset {} (word[{}] bit {})",
        aigpin, state_offset, word_idx, bit_idx
    );

    // Step 6d: Show how to set the bit
    println!(
        "\n  6d. To set port \"{}\" = 1 in global memory:",
        port_name
    );
    println!("      global_state[{}] |= (1u32 << {});", word_idx, bit_idx);
    println!("      // Or equivalently:");
    println!(
        "      global_state[{}] |= 0x{:08X};",
        word_idx,
        1u32 << bit_idx
    );

    // =========================================================================
    // STEP 7: Build a helper function for port name lookup
    // =========================================================================
    println!("\n━━━ STEP 7: Helper Function for Any Port ━━━\n");

    // Create a convenient lookup map: port_name → (aigpin, state_offset)
    let port_to_memory: IndexMap<String, (usize, u32)> = netlistdb
        .portname2pinid
        .iter()
        .filter_map(|((name, idx), &pin_id)| {
            let aigpin_iv = aig.pin2aigpin_iv[pin_id];
            let aigpin = aigpin_iv >> 1;

            // Check if this is an input
            if let Some(&offset) = script.input_map.get(&aigpin) {
                let full_name = match idx {
                    Some(i) => format!("{}[{}]", name, i),
                    None => name.to_string(),
                };
                Some((full_name, (aigpin, offset)))
            } else {
                None
            }
        })
        .collect();

    println!("  Complete port → memory mapping for inputs:");
    for (port_name, (aigpin, offset)) in &port_to_memory {
        let word_idx = offset / 32;
        let bit_idx = offset % 32;
        println!(
            "    \"{}\" → AIG pin {} → word[{}] bit {}",
            port_name, aigpin, word_idx, bit_idx
        );
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Port Name → NetlistDB.portname2pinid → pin_id                    ║");
    println!("║ pin_id → AIG.pin2aigpin_iv → aigpin (with invert bit)            ║");
    println!("║ aigpin → FlattenedScriptV1.input_map → state_offset              ║");
    println!("║ state_offset → global_state[offset/32] bit (offset%32)           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
