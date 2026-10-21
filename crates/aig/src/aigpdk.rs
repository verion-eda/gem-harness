// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! AIGPDK is a special artificial cell library used in GEM.

use compact_str::CompactString;
use netlistdb::{Direction, LeafPinProvider};
use sverilogparse::SVerilogRange;

/// This implements direction and width providers for
/// AIG PDK cells.
///
/// You can use it in netlistdb construction.
pub struct AIGPDKLeafPins();

/// Parsed DPI cell type name.
///
/// Cell names follow `$__DPI__<func>$I<w0>_<w1>_...$O<w0>_<w1>_...`.
/// This struct caches the parsed function name and per-port widths so
/// callers don't need to re-parse the encoding.
pub struct DpiCellName<'a> {
    func_name: &'a str,
    input_widths: Vec<usize>,
    output_widths: Vec<usize>,
}

impl<'a> DpiCellName<'a> {
    /// Parse a cell type name. Returns `None` if it does not start with
    /// `$__DPI__`.
    pub fn parse(macro_name: &'a str) -> Option<Self> {
        let after = macro_name.strip_prefix("$__DPI__")?;
        let (func_name, input_widths, output_widths) = match after.find("$I") {
            Some(i_pos) => {
                let func_name = &after[..i_pos];
                let after_i = &after[i_pos + 2..];
                let (iw, ow) = match after_i.find("$O") {
                    Some(pos) => (
                        Self::parse_widths(&after_i[..pos]),
                        Self::parse_widths(&after_i[pos + 2..]),
                    ),
                    None => (vec![], vec![]),
                };
                (func_name, iw, ow)
            }
            None => (after, vec![], vec![]),
        };
        Some(Self {
            func_name,
            input_widths,
            output_widths,
        })
    }

    fn parse_widths(s: &str) -> Vec<usize> {
        if s.is_empty() {
            return vec![];
        }
        s.split('_')
            .filter_map(|w| w.parse::<usize>().ok())
            .collect()
    }

    /// The pure function name (e.g. `"sat_add"`).
    pub fn func_name(&self) -> &str {
        self.func_name
    }

    /// Per-input-port widths. Empty for legacy (no `$I`/`$O` encoding).
    pub fn input_widths(&self) -> &[usize] {
        &self.input_widths
    }

    /// Per-output-port widths. Empty for legacy (no `$I`/`$O` encoding).
    pub fn output_widths(&self) -> &[usize] {
        &self.output_widths
    }

    /// Direction of a DPI pin by name (`IN_*` → input, `OUT_*` → output).
    pub fn pin_direction(pin: &str) -> Direction {
        if pin.starts_with("IN_") {
            Direction::I
        } else if pin.starts_with("OUT_") {
            Direction::O
        } else {
            panic!("Cannot recognize DPI pin direction for port {pin}")
        }
    }

    /// Width of a DPI pin, respecting the encoded widths when present.
    /// Returns `None` for single-bit pins (width <= 1).
    pub fn pin_width(&self, pin: &str) -> Option<SVerilogRange> {
        let (port_idx, widths) = if let Some(stripped) = pin.strip_prefix("IN_")
        {
            (stripped.parse::<usize>().unwrap(), &self.input_widths)
        } else if let Some(stripped) = pin.strip_prefix("OUT_") {
            (stripped.parse::<usize>().unwrap(), &self.output_widths)
        } else {
            panic!("pin needs to start with 'IN_' or 'OUT_': {pin}");
        };

        let width = if widths.is_empty() {
            32
        } else {
            widths[port_idx]
        };

        if width <= 1 {
            // TODO why dont we return SVerilogRange here, e.g.
            // SVerilogRange::empty()? Is this a convention of the
            // caller (width_of)?
            None
        } else {
            Some(SVerilogRange((width - 1) as isize, 0))
        }
    }
}

/// The addr width of an SRAM.
///
/// The word width is always 32.
/// If you change this, make sure to change all other occurences in this
/// project as well as the definitions in PDK libraries.
pub const AIGPDK_SRAM_ADDR_WIDTH: usize = 13;

pub const AIGPDK_SRAM_SIZE: usize = 1 << 13;

impl LeafPinProvider for AIGPDKLeafPins {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        pin_idx: Option<isize>,
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str(), pin_idx) {
            ("INV" | "BUF", "A", None) => Direction::I,
            ("INV" | "BUF", "Y", None) => Direction::O,

            (
                "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0"
                | "AND2_11_1",
                "A" | "B",
                None,
            ) => Direction::I,
            (
                "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0"
                | "AND2_11_1",
                "Y",
                None,
            ) => Direction::O,

            ("DFF" | "LATCH", "CLK" | "D", None) => Direction::I,
            ("DFFSR", "CLK" | "D" | "S" | "R", None) => Direction::I,
            ("DFF" | "DFFSR" | "LATCH", "Q", None) => Direction::O,

            ("CKLNQD", "CP" | "E", None) => Direction::I,
            ("CKLNQD", "Q", None) => Direction::O,

            ("$__RAMGEM_ASYNC_", _, _) => {
                panic!(
                    "Async RAM (lib cell {}) not supported yet in GEM.",
                    macro_name
                );
            }

            ("$__RAMGEM_SYNC_", "PORT_R_CLK" | "PORT_W_CLK", None) => {
                Direction::I
            }
            (
                "$__RAMGEM_SYNC_",
                "PORT_R_ADDR" | "PORT_W_ADDR",
                Some(0..=12),
            ) => Direction::I,
            (
                "$__RAMGEM_SYNC_",
                "PORT_W_WR_EN" | "PORT_W_WR_DATA",
                Some(0..=31),
            ) => Direction::I,
            ("$__RAMGEM_SYNC_", "PORT_R_RD_DATA", Some(0..=31)) => Direction::O,

            (name, pin, _) if name.starts_with("$__DPI__") => {
                DpiCellName::pin_direction(pin)
            }

            _ => {
                use netlistdb::{GeneralPinName, HierName};
                panic!(
                    "Cannot recognize pin type {}, please make sure the verilog netlist is synthesized in GEM's aigpdk.",
                    (HierName::single(macro_name.clone()), pin_name, pin_idx)
                        .dbg_fmt_pin()
                );
            }
        }
    }

    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
    ) -> Option<SVerilogRange> {
        match (macro_name.as_str(), pin_name.as_str()) {
            ("INV" | "BUF", "A" | "Y") => None,
            (
                "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0"
                | "AND2_11_1",
                "A" | "B" | "Y",
            ) => None,
            ("DFF" | "DFFSR" | "LATCH", "CLK" | "D" | "Q" | "S" | "R") => None,
            ("CKLNQD", "CP" | "E" | "Q") => None,
            ("$__RAMGEM_SYNC_", "PORT_R_CLK" | "PORT_W_CLK") => None,
            ("$__RAMGEM_SYNC_", "PORT_R_ADDR" | "PORT_W_ADDR") => {
                Some(SVerilogRange(12, 0))
            }
            (
                "$__RAMGEM_SYNC_",
                "PORT_W_WR_EN" | "PORT_W_WR_DATA" | "PORT_R_RD_DATA",
            ) => Some(SVerilogRange(31, 0)),
            (name, pin) if name.starts_with("$__DPI__") => {
                match DpiCellName::parse(name) {
                    Some(cell_name) => cell_name.pin_width(pin),
                    None => panic!("failed to parse DPI name: {name}"),
                }
            }
            _ => None,
        }
    }
}
