// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! And-Inverter Graph: the IR every later stage operates on.
//!
//! An [`AIG`] is a netlist of two-input AND gates and inverters, plus
//! "endpoint" nodes for stateful elements ([`DFF`], [`RAMBlock`]) and DPI
//! calls ([`DPICallBlock`]). The Rust datastructure is built by parsing
//! gate-level Verilog that has been synthesized against the AIG-PDK standard
//! cell library by a logic synthesizer; see the [`aigpdk`] module for the cell
//! definitions Yosys targets.
//!
//! This crate is upstream of the entire compiler pipeline: [`staging`]
//! splits an `AIG` into vertical stages, [`repcut`] partitions each stage,
//! and [`flatten`] lowers the result into a GPU bitstream. Most public
//! types here appear by reference in those crates' APIs, so changes to
//! the AIG surface ripple through the workspace.
//!
//! # Key types
//!
//! - [`AIG`] — the graph itself; nodes, edges, and endpoint tables.
//! - [`DriverType`] — what produces a signal (gate, DFF, RAM port, …).
//! - [`EndpointGroup`] — iterator over inputs of stateful endpoints.
//!
//! [`staging`]: ../staging/
//! [`repcut`]: ../repcut/
//! [`flatten`]: ../flatten/

pub mod aigpdk;

use indexmap::{IndexMap, IndexSet};
use netlistdb::{Direction, GeneralPinName, NetlistDB};

use crate::aigpdk::AIGPDK_SRAM_ADDR_WIDTH;

/// A DFF.
#[derive(Debug, Default, Clone)]
pub struct DFF {
    /// The D input pin with invert (last bit)
    pub d_iv: usize,
    /// If the DFF is enabled, i.e., if the clock, S, or R is active.
    pub en_iv: usize,
    /// The Q pin output with invert.
    pub q: usize,
}

/// A ram block resembling the interface of `$__RAMGEM_SYNC_`.
#[derive(Debug, Default, Clone)]
pub struct RAMBlock {
    pub port_r_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],

    /// controls whether r_rd_data should update. (from read clock)
    pub port_r_en_iv: usize,
    pub port_r_rd_data: [usize; 32],

    pub port_w_addr_iv: [usize; AIGPDK_SRAM_ADDR_WIDTH],
    /// controls whether memory should be updated.
    ///
    /// this is a combination of write enable and write clock.
    pub port_w_wr_en_iv: [usize; 32],
    pub port_w_wr_data_iv: [usize; 32],
}

/// A DPI call block describing a single `import "DPI-C"` function instance.
///
/// Each DPI cell (`$__DPI__<func>$I<w0>_...$O<w0>_...`) is modelled as a
/// blackbox with variable-width input ports (`IN_n`) and output ports
/// (`OUT_n`). The output ports feed the AIG as primary inputs ,
/// and the input ports are computed combinational endpoints.
#[derive(Debug, Default, Clone)]
pub struct DPICallBlock {
    /// The C function name (stripped of `$__DPI__` prefix and width encoding).
    pub func_name: String,
    /// `inputs_iv[port][bit]` = aigpin_iv of `bit` of input port
    pub inputs_iv: Vec<Vec<usize>>,
    /// `outputs[port][bit]` = aigpin (no invert) of bit `bit` of output port
    pub outputs: Vec<Vec<usize>>,
    /// Per-input-port bit widths (for multi-word callback packing).
    pub input_widths: Vec<usize>,
    /// Per-output-port bit widths (for multi-word callback packing).
    pub output_widths: Vec<usize>,
}

/// A type of endpoint group. can be a primary output-related pin,
/// a D flip-flop, or a ram block.
///
/// A group means a task for the partition to complete.
/// For primary output pins, the task is just to store.
/// For DFFs, the task is to store only when the clock is enable.
/// For RAMBlocks, the task is to simulate a sync SRAM.
/// A StagedIOPin indicates a temporary live pin between different
/// major stages but reside in the same simulated cycle.
#[derive(Debug, Copy, Clone)]
pub enum EndpointGroup<'i> {
    // TODO should be NonZeroUsize ?
    PrimaryOutput(usize),
    DFF(&'i DFF),
    RAMBlock(&'i RAMBlock),
    // TODO should be NonZeroUsize ?
    StagedIOPin(usize),
}

impl<'i> EndpointGroup<'i> {
    /// Enumerate all related aigpin inputs for this endpoint group.
    ///
    /// The enumerated inputs may have duplicates.
    pub fn for_each_input(self, mut f_nz: impl FnMut(usize)) {
        let mut f = |i| {
            if i >= 1 {
                f_nz(i);
            }
        };
        match self {
            Self::PrimaryOutput(idx) => f(idx >> 1),
            Self::DFF(dff) => {
                f(dff.en_iv >> 1);
                f(dff.d_iv >> 1);
            }
            Self::RAMBlock(ram) => {
                f(ram.port_r_en_iv >> 1);
                for i in 0..13 {
                    f(ram.port_r_addr_iv[i] >> 1);
                    f(ram.port_w_addr_iv[i] >> 1);
                }
                for i in 0..32 {
                    f(ram.port_w_wr_en_iv[i] >> 1);
                    f(ram.port_w_wr_data_iv[i] >> 1);
                }
            }
            Self::StagedIOPin(idx) => f(idx),
        }
    }

    /// Returns an iterator over all related aigpin inputs for this endpoint
    /// group.
    ///
    /// The iterator may yield duplicates. Zero inputs are filtered out.
    pub fn inputs(self) -> EndpointInputIter<'i> {
        EndpointInputIter {
            endpoint: self,
            state: 0,
        }
    }

    /// Returns a Vec of all related aigpin inputs for this endpoint group.
    ///
    /// The returned Vec may contain duplicates. Zero inputs are filtered out.
    ///
    /// This allocates, but since endpoints have at most 100 inputs (RAMBlock),
    /// the overhead is negligible.
    pub fn inputs_vec(self) -> Vec<usize> {
        let mut items = Vec::new();
        self.for_each_input(|i| items.push(i));
        items
    }
}

/// Iterator over aigpin inputs of an [`EndpointGroup`].
///
/// Created by [`EndpointGroup::inputs`].
pub struct EndpointInputIter<'i> {
    endpoint: EndpointGroup<'i>,
    state: u8,
}

impl Iterator for EndpointInputIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let val = match self.endpoint {
                EndpointGroup::PrimaryOutput(idx) => {
                    if self.state == 0 {
                        self.state = 1;
                        idx >> 1
                    } else {
                        return None;
                    }
                }
                EndpointGroup::DFF(dff) => match self.state {
                    0 => {
                        self.state = 1;
                        dff.en_iv >> 1
                    }
                    1 => {
                        self.state = 2;
                        dff.d_iv >> 1
                    }
                    _ => return None,
                },
                EndpointGroup::RAMBlock(ram) => {
                    // State encoding (matches for_each_input order):
                    // 0: port_r_en_iv
                    // 1-26: alternating port_r_addr_iv[i], port_w_addr_iv[i]
                    // for i in 0..13 27-90: alternating
                    // port_w_wr_en_iv[i], port_w_wr_data_iv[i] for i in 0..32
                    let st = self.state as usize;
                    self.state += 1;
                    match st {
                        0 => ram.port_r_en_iv >> 1,
                        1..=26 => {
                            let idx = (st - 1) / 2;
                            if (st - 1).is_multiple_of(2) {
                                ram.port_r_addr_iv[idx] >> 1
                            } else {
                                ram.port_w_addr_iv[idx] >> 1
                            }
                        }
                        27..=90 => {
                            let idx = (st - 27) / 2;
                            if (st - 27).is_multiple_of(2) {
                                ram.port_w_wr_en_iv[idx] >> 1
                            } else {
                                ram.port_w_wr_data_iv[idx] >> 1
                            }
                        }
                        _ => return None,
                    }
                }
                EndpointGroup::StagedIOPin(idx) => {
                    if self.state == 0 {
                        self.state = 1;
                        idx
                    } else {
                        return None;
                    }
                }
            };
            // Filter out zero values (same as for_each_input)
            if val >= 1 {
                return Some(val);
            }
        }
    }
}

/// The driver type of an AIG pin.
#[derive(Debug, Clone)]
pub enum DriverType {
    /// Driven by an AND gate.
    ///
    /// The inversion bit is stored as the last bits in
    /// two input indices.
    ///
    /// Only this type has combinational fan-in.
    AndGate(usize, usize),
    /// Driven by a primary input port (with its netlistdb id).
    InputPort(usize),
    /// Driven by a clock flag (with clock port netlistdb id, and pos/negedge)
    InputClockFlag(usize, u8),
    /// Driven by a DFF (with its index)
    DFF(usize),
    /// Driven by a 13-bit by 32-bit RAM block (with its index)
    SRAM(usize),
    /// Driven by a DPI call output (with the cell id of the `$__DPI__*` cell).
    ///
    /// The actual value is written by the host callback before each step.
    DPIOutput(usize),
    /// Tie0: tied to zero. Only the 0-th aig pin is allowed to have this.
    Tie0,
}

/// An AIG associated with a netlistdb.
#[derive(Debug, Default)]
pub struct AIG {
    /// The number of AIG pins.
    ///
    /// This number might be smaller than num_pins in netlistdb,
    /// because inverters and buffers are merged when possible.
    /// It might also be larger because we may add mux circuits.
    ///
    /// AIG pins are numbered from 1 to num_aigpins inclusive.
    /// The AIG pin id zero (0) is tied to 0.
    ///
    /// AIG pins are guaranteed to have topological order.
    pub num_aigpins: usize,
    /// The mapping from a netlistdb pin to an AIG pin.
    /// `usize::MAX` is a means "no AIG node for this pin".
    ///
    /// The inversion bit is stored as the last bit.
    /// E.g., `pin2aigpin_iv[pin_id] = aigpin_id << 1 | invert`.
    pub pin2aigpin_iv: Vec<usize>,
    /// The clock pins map. Every clock pin has a pair of flag pins
    /// showing if they are posedge/negedge.
    ///
    /// The flag pin can be empty which means the circuit is not
    /// active with that edge.
    pub clock_pin2aigpins: IndexMap<usize, (usize, usize)>,
    /// The driver types of AIG pins.
    pub drivers: Vec<DriverType>,
    /// A cache for identical AND gates.
    pub and_gate_cache: IndexMap<(usize, usize), usize>,
    /// Unique primary output aigpin indices
    pub primary_outputs: IndexSet<usize>,
    /// The D flip-flops (DFFs), indexed by cell id
    pub dffs: IndexMap<usize, DFF>,
    /// The SRAMs, indexed by cell id
    pub srams: IndexMap<usize, RAMBlock>,
    /// The DPI call blocks, indexed by cell id
    pub dpi_calls: IndexMap<usize, DPICallBlock>,
    /// The fanout CSR start array.
    pub fanouts_start: Vec<usize>,
    /// The fanout CSR array.
    pub fanouts: Vec<usize>,
}

impl AIG {
    fn add_aigpin(&mut self, driver: DriverType) -> usize {
        self.num_aigpins += 1;
        self.drivers.push(driver);
        self.num_aigpins
    }

    fn add_and_gate(&mut self, a: usize, b: usize) -> usize {
        assert_ne!(a | 1, usize::MAX);
        assert_ne!(b | 1, usize::MAX);
        if a == 0 || b == 0 {
            return 0;
        }
        if a == 1 {
            return b;
        }
        if b == 1 {
            return a;
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if let Some(o) = self.and_gate_cache.get(&(a, b)) {
            return o << 1;
        }
        let aigpin = self.add_aigpin(DriverType::AndGate(a, b));
        self.and_gate_cache.insert((a, b), aigpin);
        aigpin << 1
    }

    /// given a clock pin, trace back to clock root and return its
    /// enable signal (with invert bit).
    ///
    /// if result is 0, that means the pin is dangled.
    /// if an error occurs because of a undecipherable multi-input cell,
    /// we will return in error the last output pin index of that cell.
    fn trace_clock_pin(
        &mut self,
        netlistdb: &NetlistDB,
        pinid: usize,
        is_negedge: bool,
        // should we ignore cklnqd in this tracing.
        // if set to true, we will treat cklnqd as a simple buffer.
        // otherwise, we assert that cklnqd/en is already built in
        // our aig mapping (pin2aigpin_iv).
        ignore_cklnqd: bool,
    ) -> Result<usize, usize> {
        if netlistdb.pindirect[pinid] == Direction::I {
            let netid = netlistdb.pin2net[pinid];
            if Some(netid) == netlistdb.net_zero
                || Some(netid) == netlistdb.net_one
            {
                return Ok(0);
            }
            let root = netlistdb.net2pin.items[netlistdb.net2pin.start[netid]];
            return self.trace_clock_pin(
                netlistdb,
                root,
                is_negedge,
                ignore_cklnqd,
            );
        }
        let cellid = netlistdb.pin2cell[pinid];
        if cellid == 0 {
            let clkentry = self
                .clock_pin2aigpins
                .entry(pinid)
                .or_insert((usize::MAX, usize::MAX));
            let clksignal = match is_negedge {
                false => clkentry.0,
                true => clkentry.1,
            };
            if clksignal != usize::MAX {
                return Ok(clksignal << 1);
            }
            let aigpin = self.add_aigpin(DriverType::InputClockFlag(
                pinid,
                is_negedge as u8,
            ));
            let clkentry = self.clock_pin2aigpins.get_mut(&pinid).unwrap();
            let clksignal = match is_negedge {
                false => &mut clkentry.0,
                true => &mut clkentry.1,
            };
            *clksignal = aigpin;
            return Ok(aigpin << 1);
        }
        let mut pin_a = usize::MAX;
        let mut pin_cp = usize::MAX;
        let mut pin_en = usize::MAX;
        let celltype = netlistdb.celltypes[cellid].as_str();
        if !matches!(celltype, "INV" | "BUF" | "CKLNQD") {
            tracing::error!(
                "cell type {} supported on clock path. expecting only INV, BUF, or CKLNQD",
                celltype
            );
            return Err(pinid);
        }
        for ipin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pindirect[ipin] == Direction::I {
                match netlistdb.pinnames[ipin].1.as_str() {
                    "A" => pin_a = ipin,
                    "CP" => pin_cp = ipin,
                    "E" => pin_en = ipin,
                    i => {
                        tracing::error!(
                            "input pin {} unexpected for ck element {}",
                            i,
                            celltype
                        );
                        return Err(ipin);
                    }
                }
            }
        }
        match celltype {
            "INV" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb,
                    pin_a,
                    !is_negedge,
                    ignore_cklnqd,
                )
            }
            "BUF" => {
                assert_ne!(pin_a, usize::MAX);
                self.trace_clock_pin(
                    netlistdb,
                    pin_a,
                    is_negedge,
                    ignore_cklnqd,
                )
            }
            "CKLNQD" => {
                assert_ne!(pin_cp, usize::MAX);
                assert_ne!(pin_en, usize::MAX);
                let ck_iv = self.trace_clock_pin(
                    netlistdb,
                    pin_cp,
                    is_negedge,
                    ignore_cklnqd,
                )?;
                if ignore_cklnqd {
                    return Ok(ck_iv);
                }
                let en_iv = self.pin2aigpin_iv[pin_en];
                assert_ne!(en_iv, usize::MAX, "clken not built");
                Ok(self.add_and_gate(ck_iv, en_iv))
            }
            _ => unreachable!(),
        }
    }

    /// recursively add aig pins for netlistdb pins
    ///
    /// for sequential logics like DFF and RAM,
    /// 1. their netlist pin inputs are not patched,
    /// 2. their aig pin inputs (in dffs and srams arrays) will be patched to
    ///    include mux -- but not inside this function.
    /// 3. their netlist/aig outputs are directly built here, with possible
    ///    patches for asynchronous DFFSR polyfill.
    fn build_dff(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize,
        cellid: usize,
    ) {
        let q = self.add_aigpin(DriverType::DFF(cellid));
        let dff = self.dffs.entry(cellid).or_default();
        dff.q = q;
        let mut ap_s_iv = 1;
        let mut ap_r_iv = 1;
        let mut q_out = q << 1;
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            if !matches!(netlistdb.pinnames[pinid].1.as_str(), "S" | "R") {
                continue;
            }
            self.dfs_netlistdb_build_aig(
                netlistdb,
                topo_vis,
                topo_instack,
                pinid,
            );
            let prev = self.pin2aigpin_iv[pinid];
            match netlistdb.pinnames[pinid].1.as_str() {
                "S" => ap_s_iv = prev,
                "R" => ap_r_iv = prev,
                _ => unreachable!(),
            }
        }
        q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
        q_out = self.add_and_gate(q_out, ap_r_iv);
        self.pin2aigpin_iv[pinid] = q_out;
    }

    fn build_sram(
        &mut self,
        netlistdb: &NetlistDB,
        pinid: usize,
        cellid: usize,
    ) {
        let o = self.add_aigpin(DriverType::SRAM(cellid));
        self.pin2aigpin_iv[pinid] = o << 1;
        assert_eq!(netlistdb.pinnames[pinid].1.as_str(), "PORT_R_RD_DATA");
        let sram = self.srams.entry(cellid).or_default();
        sram.port_r_rd_data[netlistdb.pinnames[pinid].2.unwrap() as usize] = o;
    }

    fn build_dpi(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize,
        cellid: usize,
    ) {
        // Direction::O (OUT_n pin): create a DPIOutput aigpin.
        let o = self.add_aigpin(DriverType::DPIOutput(cellid));
        self.pin2aigpin_iv[pinid] = o << 1;
        // DPIOutput is a leaf in the AIG (like DFF Q), so the DFS
        // won't naturally recurse into this cell's IN_n pins.  We must
        // explicitly resolve them here so that pin2aigpin_iv is populated
        // before the post-pass builds DPICallBlock.inputs_iv.
        for in_pin in netlistdb.cell2pin.iter_set(cellid) {
            if netlistdb.pinnames[in_pin].1.as_str().starts_with("IN_") {
                self.dfs_netlistdb_build_aig(
                    netlistdb,
                    topo_vis,
                    topo_instack,
                    in_pin,
                );
            }
        }
    }

    fn build_cklnqd(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        cellid: usize,
    ) {
        let mut prev_cp = usize::MAX;
        let mut prev_en = usize::MAX;
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            match netlistdb.pinnames[pinid].1.as_str() {
                "CP" => prev_cp = pinid,
                "E" => prev_en = pinid,
                _ => {}
            }
        }
        assert_ne!(prev_cp, usize::MAX);
        assert_ne!(prev_en, usize::MAX);
        for prev in [prev_cp, prev_en] {
            self.dfs_netlistdb_build_aig(
                netlistdb,
                topo_vis,
                topo_instack,
                prev,
            );
        }
        // do not define pin2aigpin_iv[pinid] which is CKLNQD/Q and unused
        // in logic.
    }

    fn build_logic_gate(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize,
        cellid: usize,
        celltype: &str,
    ) {
        let mut prev_a = usize::MAX;
        let mut prev_b = usize::MAX;
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            match netlistdb.pinnames[pinid].1.as_str() {
                "A" => prev_a = pinid,
                "B" => prev_b = pinid,
                _ => {}
            }
        }
        for prev in [prev_a, prev_b] {
            if prev != usize::MAX {
                self.dfs_netlistdb_build_aig(
                    netlistdb,
                    topo_vis,
                    topo_instack,
                    prev,
                );
            }
        }
        match celltype {
            "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0"
            | "AND2_11_1" => {
                assert_ne!(prev_a, usize::MAX);
                assert_ne!(prev_b, usize::MAX);
                let name = netlistdb.celltypes[cellid].as_bytes();
                let iv_a = name[5] - b'0';
                let iv_b = name[6] - b'0';
                let iv_y = name[8] - b'0';
                let apid = self.add_and_gate(
                    self.pin2aigpin_iv[prev_a] ^ (iv_a as usize),
                    self.pin2aigpin_iv[prev_b] ^ (iv_b as usize),
                ) ^ (iv_y as usize);
                self.pin2aigpin_iv[pinid] = apid;
            }
            "INV" => {
                assert_ne!(prev_a, usize::MAX);
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a] ^ 1;
            }
            "BUF" => {
                assert_ne!(prev_a, usize::MAX);
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a];
            }
            _ => unreachable!(),
        }
    }

    fn dfs_netlistdb_build_aig(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize,
    ) {
        if topo_instack[pinid] {
            panic!(
                "circuit has a loop around pin {}",
                netlistdb.pinnames[pinid].dbg_fmt_pin()
            );
        }
        if topo_vis[pinid] {
            return;
        }
        topo_vis[pinid] = true;
        topo_instack[pinid] = true;
        let netid = netlistdb.pin2net[pinid];
        let cellid = netlistdb.pin2cell[pinid];
        let celltype = netlistdb.celltypes[cellid].as_str();

        if netlistdb.pindirect[pinid] == Direction::I {
            if Some(netid) == netlistdb.net_zero {
                self.pin2aigpin_iv[pinid] = 0;
            } else if Some(netid) == netlistdb.net_one {
                self.pin2aigpin_iv[pinid] = 1;
            } else {
                let root =
                    netlistdb.net2pin.items[netlistdb.net2pin.start[netid]];
                self.dfs_netlistdb_build_aig(
                    netlistdb,
                    topo_vis,
                    topo_instack,
                    root,
                );
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[root];
                if cellid == 0 {
                    self.primary_outputs.insert(self.pin2aigpin_iv[pinid]);
                }
            }
            topo_instack[pinid] = false;
            return;
        }

        if cellid == 0 {
            let aigpin = self.add_aigpin(DriverType::InputPort(pinid));
            self.pin2aigpin_iv[pinid] = aigpin << 1;
            topo_instack[pinid] = false;
            return;
        }

        match celltype {
            "DFF" | "DFFSR" => {
                self.build_dff(netlistdb, topo_vis, topo_instack, pinid, cellid)
            }
            "LATCH" => panic!(
                "latches are intentionally UNSUPPORTED by GEM, \
                    except in identified gated clocks. \n\
                    you can link a FF&MUX-based LATCH module, \
                    but most likely that is NOT the right solution. \n\
                    check all your assignments inside always@(*) block \
                    to make sure they cover all scenarios."
            ),
            "$__RAMGEM_SYNC_" => self.build_sram(netlistdb, pinid, cellid),
            ct if ct.starts_with("$__DPI__") => {
                self.build_dpi(netlistdb, topo_vis, topo_instack, pinid, cellid)
            }
            "CKLNQD" => {
                self.build_cklnqd(netlistdb, topo_vis, topo_instack, cellid)
            }
            _ => self.build_logic_gate(
                netlistdb,
                topo_vis,
                topo_instack,
                pinid,
                cellid,
                celltype,
            ),
        }

        topo_instack[pinid] = false;
    }

    /// Converts a [`NetlistDB`] gate-level netlist into the [`AIG`] graph
    /// representation.
    ///
    /// The input netlist is already an AIG — Yosys has synthesized the RTL
    /// against the `aigpdk` cell library (AND2X0, INVX0, DFF/DFFSR, RAMGEM,
    /// clock gates). This function re-encodes that flat netlist into the
    /// compact [`AIG`] data structure used by all downstream stages (staging,
    /// RepCut partitioning, boomerang-layer placement, and bitstream
    /// generation), folding inverters into inversion bits and eliding buffers
    /// rather than creating explicit nodes for either.
    ///
    /// The construction proceeds in five phases:
    ///
    /// 1. **Clock inference** — Traces clock pins on all sequential elements
    ///    (DFF, DFFSR, RAMGEM) back through buffers and inverters to identify
    ///    clock domains and their active edges (posedge / negedge).
    ///
    /// 2. **Combinational DFS** — Visits every netlist pin and recursively
    ///    builds the AIG graph via [`dfs_netlistdb_build_aig`], assigning each
    ///    pin an `aigpin_iv` (node index with inversion bit in the LSB).
    ///    Inverters are folded into the inversion bit and buffers are elided,
    ///    so neither creates a dedicated AIG node.
    ///
    /// 3. **Endpoint registration** — Marks signals that must be observable in
    ///    the GPU state vector as primary outputs: `keep`-annotated nets and
    ///    every bit of every DPI input argument (so the CPU can read them back
    ///    to assemble DPI call arguments at runtime). Without this, their
    ///    combinational cones would be pruned from the boomerang layers.
    ///
    /// 4. **Sequential element wiring** — Connects DFF / DFFSR data, set,
    ///    reset, and clock-enable inputs, and wires SRAM read/write address,
    ///    data, and enable ports (including gating write-enables with the write
    ///    clock).
    ///
    /// 5. **Fanout index construction** — Builds a CSR (Compressed Sparse Row)
    ///    fanout array over all AND gates so downstream passes can efficiently
    ///    walk the graph in the forward direction.
    fn wire_dff(&mut self, netlistdb: &NetlistDB, cellid: usize) {
        let mut ap_s_iv = 1;
        let mut ap_r_iv = 1;
        let mut ap_d_iv = 0;
        let mut ap_clken_iv = 0;
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            let pin_iv = self.pin2aigpin_iv[pinid];
            match netlistdb.pinnames[pinid].1.as_str() {
                "D" => ap_d_iv = pin_iv,
                "S" => ap_s_iv = pin_iv,
                "R" => ap_r_iv = pin_iv,
                "CLK" => {
                    ap_clken_iv = self
                        .trace_clock_pin(netlistdb, pinid, false, false)
                        .unwrap()
                }
                _ => {}
            }
        }
        let mut d_in = ap_d_iv;

        d_in = self.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
        ap_clken_iv = self.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
        d_in = self.add_and_gate(d_in, ap_r_iv);
        ap_clken_iv = self.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;
        let dff = self.dffs.entry(cellid).or_default();
        dff.en_iv = ap_clken_iv;
        dff.d_iv = d_in;
        assert_ne!(dff.q, 0);
    }

    fn wire_sram(&mut self, netlistdb: &NetlistDB, cellid: usize) {
        let mut sram = self.srams.entry(cellid).or_default().clone();
        let mut write_clken_iv = 0;
        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
            let pin_iv = self.pin2aigpin_iv[pinid];
            match netlistdb.pinnames[pinid].1.as_str() {
                "PORT_R_ADDR" => {
                    sram.port_r_addr_iv[bit.unwrap()] = pin_iv;
                }
                "PORT_R_CLK" => {
                    sram.port_r_en_iv = self
                        .trace_clock_pin(netlistdb, pinid, false, false)
                        .unwrap();
                }
                "PORT_W_ADDR" => {
                    sram.port_w_addr_iv[bit.unwrap()] = pin_iv;
                }
                "PORT_W_CLK" => {
                    write_clken_iv = self
                        .trace_clock_pin(netlistdb, pinid, false, false)
                        .unwrap();
                }
                "PORT_W_WR_DATA" => {
                    sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                }
                "PORT_W_WR_EN" => {
                    sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                }
                _ => {}
            }
        }
        for i in 0..32 {
            let or_en = sram.port_w_wr_en_iv[i];
            let or_en = self.add_and_gate(or_en, write_clken_iv);
            sram.port_w_wr_en_iv[i] = or_en;
        }
        *self.srams.get_mut(&cellid).unwrap() = sram;
    }

    fn wire_dpi(&mut self, netlistdb: &NetlistDB, cellid: usize) {
        let celltype = netlistdb.celltypes[cellid].as_str();
        let dpi_name = crate::aigpdk::DpiCellName::parse(celltype)
            .expect("DPI cell type must start with $__DPI__");
        let func_name = dpi_name.func_name().to_string();
        let in_widths = dpi_name.input_widths().to_vec();
        let out_widths = dpi_name.output_widths().to_vec();

        let mut in_map: IndexMap<usize, Vec<usize>> = IndexMap::new();
        let mut out_map: IndexMap<usize, Vec<usize>> = IndexMap::new();

        for pinid in netlistdb.cell2pin.iter_set(cellid) {
            let pname = netlistdb.pinnames[pinid].1.as_str();
            let bit =
                netlistdb.pinnames[pinid].2.map(|i| i as usize).unwrap_or(0);
            match crate::aigpdk::DpiCellName::pin_direction(pname) {
                netlistdb::Direction::I => {
                    let port_idx: usize = pname[3..].parse().unwrap();
                    let width = in_widths.get(port_idx).copied().unwrap_or(32);
                    let entry = in_map
                        .entry(port_idx)
                        .or_insert_with(|| vec![0; width]);
                    entry[bit] = self.pin2aigpin_iv[pinid];
                }
                netlistdb::Direction::O => {
                    let port_idx: usize = pname[4..].parse().unwrap();
                    let width = out_widths.get(port_idx).copied().unwrap_or(32);
                    let entry = out_map
                        .entry(port_idx)
                        .or_insert_with(|| vec![0; width]);
                    entry[bit] = self.pin2aigpin_iv[pinid] >> 1;
                }
                _ => unreachable!(),
            }
        }

        let num_in_ports = in_map.len();
        let num_out_ports = out_map.len();
        let inputs_iv: Vec<Vec<usize>> =
            (0..num_in_ports).map(|i| in_map[&i].clone()).collect();
        let outputs: Vec<Vec<usize>> =
            (0..num_out_ports).map(|i| out_map[&i].clone()).collect();

        let input_widths = if in_widths.is_empty() {
            vec![32; num_in_ports]
        } else {
            in_widths
        };
        let output_widths = if out_widths.is_empty() {
            vec![32; num_out_ports]
        } else {
            out_widths
        };

        // Register DPI input IVs as primary outputs so their
        // combinational cones are included in the state vector.
        for port_ivs in &inputs_iv {
            for &iv in port_ivs {
                // Skip AIG node 0 (the constant false literal)
                if (iv >> 1) != 0 {
                    self.primary_outputs.insert(iv);
                }
            }
        }

        self.dpi_calls.insert(
            cellid,
            DPICallBlock {
                func_name,
                inputs_iv,
                outputs,
                input_widths,
                output_widths,
            },
        );
    }

    pub fn from_netlistdb(netlistdb: &NetlistDB) -> AIG {
        let mut aig = AIG {
            num_aigpins: 0,
            // The AIG only models computation — it's a graph of logic gates
            // where every node represents something being computed or driven.
            // A sink pin (e.g. a primary output port) doesn't compute anything;
            //  it just observes a value computed elsewhere.
            // usize::MAX is a sentinel meaning "no AIG node for this pin".
            pin2aigpin_iv: vec![usize::MAX; netlistdb.num_pins],
            drivers: vec![DriverType::Tie0],
            ..Default::default()
        };

        for cellid in 1..netlistdb.num_cells {
            if !matches!(
                netlistdb.celltypes[cellid].as_str(),
                "DFF" | "DFFSR" | "$__RAMGEM_SYNC_"
            ) {
                continue;
            }
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(
                    netlistdb.pinnames[pinid].1.as_str(),
                    "CLK" | "PORT_R_CLK" | "PORT_W_CLK"
                ) {
                    continue;
                }
                if let Err(pinid) =
                    aig.trace_clock_pin(netlistdb, pinid, false, true)
                {
                    use netlistdb::GeneralHierName;
                    panic!(
                        "Tracing clock pin of cell {} error: \
                            there is a multi-input cell driving {} \
                            that clocks this sequential element. \
                            Clock gating need to be manually patched atm.",
                        netlistdb.cellnames[cellid].dbg_fmt_hier(),
                        netlistdb.pinnames[pinid].dbg_fmt_pin()
                    );
                }
            }
        }
        for (&clk, &(flagr, flagf)) in &aig.clock_pin2aigpins {
            tracing::info!(
                "inferred clock port {} ({})",
                netlistdb.pinnames[clk].dbg_fmt_pin(),
                match (flagr, flagf) {
                    (_, usize::MAX) => "posedge",
                    (usize::MAX, _) => "negedge",
                    _ => "posedge & negedge",
                }
            );
        }

        let mut topo_vis = vec![false; netlistdb.num_pins];
        let mut topo_instack = vec![false; netlistdb.num_pins];

        for pinid in 0..netlistdb.num_pins {
            aig.dfs_netlistdb_build_aig(
                netlistdb,
                &mut topo_vis,
                &mut topo_instack,
                pinid,
            );
        }

        // Treat keep-annotated nets as synthetic primary outputs so their
        // values are allocated a slot in the simulation state vector and
        // remain observable after flattening. The DFS above has already
        // assigned aigpin_iv values to all net drivers, so we only need to
        // register them as endpoints here.
        for &net_id in &netlistdb.keep_nets {
            let root_pin =
                netlistdb.net2pin.items[netlistdb.net2pin.start[net_id]];
            let aigpin_iv = aig.pin2aigpin_iv[root_pin];
            if aigpin_iv != usize::MAX {
                aig.primary_outputs.insert(aigpin_iv);
            }
        }

        for cellid in 0..netlistdb.num_cells {
            match netlistdb.celltypes[cellid].as_str() {
                "DFF" | "DFFSR" => aig.wire_dff(netlistdb, cellid),
                "$__RAMGEM_SYNC_" => aig.wire_sram(netlistdb, cellid),
                ct if ct.starts_with("$__DPI__") => {
                    aig.wire_dpi(netlistdb, cellid)
                }
                _ => {}
            }
        }

        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        for driver in aig.drivers.iter() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    aig.fanouts_start[a >> 1] += 1;
                }
                if (b >> 1) != 0 {
                    aig.fanouts_start[b >> 1] += 1;
                }
            }
        }
        for i in 1..aig.num_aigpins + 2 {
            aig.fanouts_start[i] += aig.fanouts_start[i - 1];
        }
        aig.fanouts = vec![0; aig.fanouts_start[aig.num_aigpins + 1]];
        for (i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    let st = aig.fanouts_start[a >> 1] - 1;
                    aig.fanouts_start[a >> 1] = st;
                    aig.fanouts[st] = i;
                }
                if (b >> 1) != 0 {
                    let st = aig.fanouts_start[b >> 1] - 1;
                    aig.fanouts_start[b >> 1] = st;
                    aig.fanouts[st] = i;
                }
            }
        }

        aig
    }

    pub fn topo_traverse_generic(
        &self,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> Vec<usize> {
        let mut vis = IndexSet::new();
        let mut ret = Vec::new();

        fn dfs_topo(
            drivers: &Vec<DriverType>,
            vis: &mut IndexSet<usize>,
            ret: &mut Vec<usize>,
            is_primary_input: Option<&IndexSet<usize>>,
            u: usize,
        ) {
            if vis.contains(&u) {
                return;
            }
            vis.insert(u);

            if let DriverType::AndGate(a, b) = drivers[u]
                && is_primary_input.map(|s| s.contains(&u)) != Some(true)
            {
                if (a >> 1) != 0 {
                    dfs_topo(drivers, vis, ret, is_primary_input, a >> 1);
                }
                if (b >> 1) != 0 {
                    dfs_topo(drivers, vis, ret, is_primary_input, b >> 1);
                }
            }

            ret.push(u);
        }

        if let Some(endpoints) = endpoints {
            for &endpoint in endpoints {
                dfs_topo(
                    &self.drivers,
                    &mut vis,
                    &mut ret,
                    is_primary_input,
                    endpoint,
                );
            }
        } else {
            for i in 1..self.num_aigpins + 1 {
                dfs_topo(
                    &self.drivers,
                    &mut vis,
                    &mut ret,
                    is_primary_input,
                    i,
                );
            }
        }

        ret
    }

    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_outputs.len() + self.dffs.len() + self.srams.len()
    }

    pub fn get_endpoint_group(&self, endpt_id: usize) -> EndpointGroup<'_> {
        if endpt_id < self.primary_outputs.len() {
            EndpointGroup::PrimaryOutput(
                *self.primary_outputs.get_index(endpt_id).unwrap(),
            )
        } else if endpt_id < self.primary_outputs.len() + self.dffs.len() {
            EndpointGroup::DFF(
                &self.dffs[endpt_id - self.primary_outputs.len()],
            )
        } else {
            EndpointGroup::RAMBlock(
                &self.srams
                    [endpt_id - self.primary_outputs.len() - self.dffs.len()],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aigpdk::AIGPDKLeafPins;

    const ADDER_WITH_KEEP: &str = r#"
module adder(a, b, sum, carry);
  input a;
  wire a;
  input b;
  wire b;
  output sum;
  wire sum;
  output carry;
  wire carry;
  (* keep *) wire _0_;
  AND2_00_0 _1_ (
    .A(b),
    .B(a),
    .Y(carry)
  );
  AND2_11_1 _2_ (
    .A(b),
    .B(a),
    .Y(_0_)
  );
  AND2_10_0 _3_ (
    .A(carry),
    .B(_0_),
    .Y(sum)
  );
endmodule
"#;

    #[test]
    fn test_keep_wire_becomes_primary_output() {
        let netlist = NetlistDB::from_sverilog_source(
            ADDER_WITH_KEEP,
            None,
            &AIGPDKLeafPins(),
        )
        .expect("failed to parse adder netlist");

        let kept_pin_ids: Vec<usize> = netlist
            .keep_nets
            .iter()
            .map(|&net_id| netlist.net2pin.items[netlist.net2pin.start[net_id]])
            .collect();
        assert_eq!(kept_pin_ids.len(), 1);
        let pin_id = kept_pin_ids.first().unwrap();

        let aig = AIG::from_netlistdb(&netlist);

        // The kept wire _0_ must have an allocated aigpin (not usize::MAX)
        // and that aigpin_iv must appear in primary_outputs.
        let aigpin_iv = aig.pin2aigpin_iv[*pin_id];
        assert_ne!(
            aigpin_iv,
            usize::MAX,
            "keep net pin {} has no aigpin assignment",
            pin_id
        );
        assert!(
            aig.primary_outputs.contains(&aigpin_iv),
            "keep net aigpin_iv {} not in primary_outputs",
            aigpin_iv
        );
    }
}
