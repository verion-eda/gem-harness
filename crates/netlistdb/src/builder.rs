// SPDX-FileCopyrightText: Copyright (c) Zizheng Guo. Originally from eda-infra-rs (https://github.com/gzz2000/eda-infra-rs).
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
// SPDX-License-Identifier: Apache-2.0
//! NetlistDB construction from parsed structural verilog.
//!
//! Building a [`NetlistDB`] is a two-step process:
//!
//! **Step 1** — [`NetlistDB::init_graph_from_modules`]: Traverses and flattens
//! the module hierarchy starting from the top-level module. Cells, logic pins,
//! and nets are allocated, and connectivity is established via a disjoint-set
//! (union-find) structure. At the end of this step the graph topology is fully
//! built: pin–cell, pin–net, and cell–net mappings are all finalized. Pin
//! directions are left unassigned.
//!
//! **Step 2** — [`NetlistDB::assign_direction`]: Fills in pin directions by
//! querying the [`LeafPinProvider`] for leaf-cell pins and reading the wire
//! declaration types (`input`/`output`) for top-level ports. After directions
//! are known, each net's CSR entry is reordered so that the single driving
//! (output) pin appears first.

use std::borrow::Cow;

use either::Either;
use rayon::prelude::*;
use sverilogparse::*;

use super::*;

fn pin_assign_literal(net_sets: &mut DisjointSet, l: usize, c: u8) {
    match c {
        0 => net_sets.set_value(l, false),
        1 => net_sets.set_value(l, true),
        2 => {
            tracing::warn!("X literal unsupported, treating as 0");
            net_sets.set_value(l, false);
        }
        3 => {
            tracing::warn!(
                "Z literal ignored (high-impedance, no net assignment)"
            );
        }
        _ => unreachable!(),
    }
}

/// Leaf pin direction and width provider trait.
///
/// Downstream databases (e.g., Liberty library or LEF/DEF library)
/// should implement this to provide pin direction information
/// used in constructing NetlistDB.
///
/// This was called DirectionProvider before. That name can still
/// be used as an alias, but deprecated.
pub trait LeafPinProvider {
    /// This function is called from NetlistDB constructor to
    /// query the direction of library cell pins.
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        pin_idx: Option<isize>,
    ) -> Direction;

    // This function is called to query the range of the bus width
    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
    ) -> Option<SVerilogRange>;

    /// This function allows downstream databases to specify
    /// whether there should be a warning on unspecified
    /// directions when building netlist.
    #[inline]
    fn should_warn_missing_directions(&self) -> bool {
        true
    }
}

#[doc(hidden)]
pub use LeafPinProvider as DirectionProvider;

impl DirectionProvider
    for HashMap<(CompactString, CompactString, Option<isize>), Direction>
{
    #[inline]
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        pin_idx: Option<isize>,
    ) -> Direction {
        let k = (macro_name.clone(), pin_name.clone(), pin_idx);
        self.get(&k).copied().unwrap_or(Direction::Unknown)
    }

    #[inline]
    fn width_of(
        &self,
        _: &CompactString,
        _: &CompactString,
    ) -> Option<SVerilogRange> {
        None
    }
}

impl<T> DirectionProvider for T
where
    T: Fn(&CompactString, &CompactString, Option<isize>) -> Direction,
{
    #[inline]
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString,
        pin_idx: Option<isize>,
    ) -> Direction {
        self(macro_name, pin_name, pin_idx)
    }

    #[inline]
    fn width_of(
        &self,
        _: &CompactString,
        _: &CompactString,
    ) -> Option<SVerilogRange> {
        None
    }
}

/// A special direction hint that gives no answer to every pin.
/// This is useful if you do not care about the pin direction
/// (e.g. if you are outputting the benchmark statistics only).
pub struct NoDirection;

impl DirectionProvider for NoDirection {
    #[inline]
    fn direction_of(
        &self,
        _: &CompactString,
        _: &CompactString,
        _: Option<isize>,
    ) -> Direction {
        Direction::Unknown
    }

    #[inline]
    fn width_of(
        &self,
        _: &CompactString,
        _: &CompactString,
    ) -> Option<SVerilogRange> {
        None
    }

    /// Directions are explicitly ignored, so disable all
    /// such warnings.
    #[inline]
    fn should_warn_missing_directions(&self) -> bool {
        false
    }
}

impl NetlistDB {
    /// Convenient shortcut to read from file.
    /// The parameters are similar to [NetlistDB::from_sverilog].
    pub fn from_sverilog_file(
        sverilog_source_path: impl AsRef<std::path::Path>,
        top: Option<&str>,
        direction_provider: &impl DirectionProvider,
    ) -> Option<NetlistDB> {
        let sverilog = match SVerilog::parse_file(&sverilog_source_path) {
            Ok(sv) => sv,
            Err(e) => {
                tracing::error!(
                    "Parse sverilog file {} failed: {}",
                    sverilog_source_path.as_ref().display(),
                    e
                );
                return None;
            }
        };
        NetlistDB::from_sverilog(sverilog, top, direction_provider)
    }

    /// Convenient shortcut to read from a source string.
    /// The parameters are similar to [NetlistDB::from_sverilog].
    pub fn from_sverilog_source(
        sverilog_source: &str,
        top: Option<&str>,
        direction_provider: &impl DirectionProvider,
    ) -> Option<NetlistDB> {
        let sverilog = match SVerilog::parse_str(sverilog_source) {
            Ok(sv) => sv,
            Err(e) => {
                tracing::error!("Parse sverilog source code failed: {}", e);
                return None;
            }
        };
        NetlistDB::from_sverilog(sverilog, top, direction_provider)
    }

    /// Build a database from a parsed structural verilog object.
    ///
    /// The top module to be built from can be optionally specified through
    /// the `top` parameter.
    ///
    /// There should be a way to specify library pin directions --
    /// through a trait called direction provider.
    pub fn from_sverilog(
        sverilog: SVerilog,
        top: Option<&str>,
        direction_provider: &impl DirectionProvider,
    ) -> Option<NetlistDB> {
        let modules: HashMap<CompactString, (SVerilogModule, ModuleMap)> =
            sverilog
                .modules
                .into_iter()
                .map(|(k, v)| {
                    let mm = ModuleMap::from(&v);
                    (k, (v, mm))
                })
                .collect();

        let top = find_top_module(&modules, top)?;

        let mut db = NetlistDB::init_graph_from_modules(
            &modules,
            top,
            direction_provider,
        )?;

        db.assign_direction(top, direction_provider)?;

        Some(db)
    }

    /// Get or insert a logic pin with hier, name and index.
    #[inline]
    fn get_or_insert_logic_pin(
        &mut self,
        hier: &HierName,
        name: &CompactString,
        idx: Option<isize>,
        keep: bool,
    ) -> usize {
        let k = (hier.clone(), name.clone(), idx);
        if let Some(&i) = self.logicpinname2id.get(&k) {
            if keep {
                self.keep_logic_pins.insert(i);
            }
            return i;
        }
        let id = self.num_logic_pins;
        self.num_logic_pins += 1;
        self.logicpinname2id.insert(k.clone(), id);
        self.logicpintypes.push(LogicPinType::Others);
        if keep {
            self.keep_logic_pins.insert(id);
        }
        self.logicpinnames.push(k);
        id
    }

    /// Find the logic pin with a specific name.
    /// If not found, *prints an error message* and returns None.
    /// This is useful in netlist construction.
    #[inline]
    #[must_use]
    fn try_find_logic_pin(
        &self,
        hier: &HierName,
        name: &CompactString,
        idx: Option<isize>,
    ) -> Option<usize> {
        let k = (hier.clone(), name.clone(), idx);
        let r = self.logicpinname2id.get(&k);
        if r.is_none() {
            tracing::error!(
                "pin/net reference {}/{} [{:?}] not found",
                hier,
                name,
                idx
            );
        }
        r.copied()
    }

    /// Insert a cell with hier name and macro name.
    /// Returns the new cell id.
    #[inline]
    fn insert_cell(
        &mut self,
        hier: HierName,
        macro_name: CompactString,
    ) -> usize {
        let id = self.num_cells;
        self.num_cells += 1;
        self.cellname2id.insert(hier.clone(), id);
        self.celltypes.push(macro_name);
        self.cellnames.push(hier);
        id
    }

    /// Recursively build (and flatten) the hierarchical modules.
    /// `net_sets` is the disjoint set of logic pins into nets.
    /// `hier` is the current hier name.
    /// This call will build all port pins inside it. So no need to
    /// build them outside.
    #[must_use]
    fn build_modules(
        &mut self,
        modules: &HashMap<CompactString, (SVerilogModule, ModuleMap)>,
        (_m_name, m, mm): (&CompactString, &SVerilogModule, &ModuleMap),
        net_sets: &mut DisjointSet,
        hier: HierName,
        lib: &impl LeafPinProvider,
        force_keep: bool,
    ) -> Option<()> {
        let effective_keep = force_keep || m.keep;

        // create nets/IO logic pins
        for def in &m.defs {
            for w in enum_in_width(def.width) {
                let id =
                    self.get_or_insert_logic_pin(&hier, &def.name, w, def.keep);
                // for the top module, the ports are tagged as
                // LogicPinType::TopPort outside the first invocation.
                // (in [NetlistDB::from_sverilog]).
                self.logicpintypes[id] = LogicPinType::Net;
            }
        }

        // create named ports
        for (name, expr) in m.ports.iter().filter_map(|p| match p {
            SVerilogPortDef::Basic(_) => None,
            SVerilogPortDef::Conn(name, expr) => Some((name, expr)),
        }) {
            let width = mm.port_widths.get(name).copied();
            for (id, eb) in enum_in_width(width).zip(mm.eval_expr(expr)) {
                let port_id =
                    self.get_or_insert_logic_pin(&hier, name, id, false);
                use ExprBit::*;
                match eb {
                    Const(c) => {
                        pin_assign_literal(net_sets, port_id, c);
                    }
                    Var(pname, pidx) => {
                        let pin_id =
                            self.try_find_logic_pin(&hier, pname, pidx)?;
                        net_sets.merge(port_id, pin_id);
                    }
                }
            }
        }

        let hier_prev = match hier.is_empty() {
            true => None,
            false => Some(Arc::new(hier.clone())),
        };

        // recurse into submodules and cells
        for cell in &m.cells {
            let new_hier = HierName {
                prev: hier_prev.clone(),
                cur: cell.cell_name.clone(),
            };

            // build submodule/cell and get ranges of ports
            let (is_leaf, ioport_ranges) = match modules.get(&cell.macro_name) {
                Some((subm, submm)) => {
                    // non-leaf.
                    self.build_modules(
                        modules,
                        (&cell.macro_name, subm, submm),
                        net_sets,
                        new_hier.clone(),
                        lib,
                        effective_keep || cell.keep,
                    )?;
                    (
                        false,
                        Either::Left(cell.ioports.iter().map(|(name, _)| {
                            submm.port_widths.get(name.as_str()).copied()
                        })),
                    )
                }
                None => {
                    // leaf cell.
                    let cell_id = self
                        .insert_cell(new_hier.clone(), cell.macro_name.clone());
                    if effective_keep || cell.keep {
                        self.keep_cells.insert(cell_id);
                    }
                    (
                        true,
                        Either::Right(cell.ioports.iter().map(
                            |(macro_pin_name, expr)| {
                                // mimics CadXX InnoXX:
                                // a leaf cell port is a scalar iff width == 1.
                                match mm.eval_expr_len(expr) {
                                    1 => None,
                                    _ => {
                                        match lib.width_of(
                                            &cell.macro_name,
                                            macro_pin_name,
                                        ) {
                                            Some(SVerilogRange(
                                                left,
                                                right,
                                            )) => {
                                                Some(SVerilogRange(left, right))
                                            }
                                            _ => None,
                                        }
                                    }
                                }
                            },
                        )),
                    )
                }
            };

            // connect edges.
            for (w, (name, expr)) in ioport_ranges.zip(cell.ioports.iter()) {
                for (i, eb) in enum_in_width(w).zip(mm.eval_expr(expr)) {
                    let id = match is_leaf {
                        // if it is a leaf, we insert the pin.
                        true => self
                            .get_or_insert_logic_pin(&new_hier, name, i, false),
                        // if it is a submodule, the pin should already be
                        // ready, so we assert its
                        // existence.
                        false => self.try_find_logic_pin(&new_hier, name, i)?,
                    };
                    if is_leaf {
                        self.logicpintypes[id] = LogicPinType::LeafCellPin;
                    }
                    match eb {
                        ExprBit::Const(c) => {
                            pin_assign_literal(net_sets, id, c);
                        }
                        ExprBit::Var(pname, pidx) => {
                            // a wire might be used but not defined.
                            let eb_id = self.get_or_insert_logic_pin(
                                &hier, pname, pidx, false,
                            );
                            let typ = &mut self.logicpintypes[eb_id];
                            if *typ == LogicPinType::Others {
                                *typ = LogicPinType::Net;
                            }
                            net_sets.merge(id, eb_id);
                        }
                    }
                }
            }
        }

        // connect assignments
        for assign in &m.assigns {
            let len_lhs = mm.eval_expr_len(&assign.lhs);
            let len_rhs = mm.eval_expr_len(&assign.rhs);
            if len_lhs != len_rhs {
                tracing::error!(
                    "incompatible assign width for `{}`: \
                     len(LHS) = {}, len(RHS) = {}",
                    assign,
                    len_lhs,
                    len_rhs
                );
                return None;
            }

            for (lb, rb) in
                mm.eval_expr(&assign.lhs).zip(mm.eval_expr(&assign.rhs))
            {
                match (lb, rb) {
                    (ExprBit::Var(nl, il), ExprBit::Var(nr, ir)) => {
                        let l = self.get_or_insert_logic_pin(
                            &hier,
                            nl,
                            il,
                            assign.keep,
                        );
                        let r = self.get_or_insert_logic_pin(
                            &hier,
                            nr,
                            ir,
                            assign.keep,
                        );
                        net_sets.merge(l, r);
                    }
                    (ExprBit::Var(nl, il), ExprBit::Const(c)) => {
                        let l = self.get_or_insert_logic_pin(
                            &hier,
                            nl,
                            il,
                            assign.keep,
                        );
                        pin_assign_literal(net_sets, l, c);
                    }
                    (ExprBit::Const(c), ExprBit::Var(nr, ir)) => {
                        let r = self.get_or_insert_logic_pin(
                            &hier,
                            nr,
                            ir,
                            assign.keep,
                        );
                        pin_assign_literal(net_sets, r, c);
                    }
                    _ => {
                        tracing::error!("Bad lit-lit assign.");
                        return None;
                    }
                }
            }
        }

        Some(())
    }

    /// Building a netlist database STEP 1: initialize most of the
    /// graph structure using parsed verilog modules starting from
    /// the top-level module.
    ///
    /// The only remaining thing to do is to assign pin directions.
    ///
    /// Splitting this out would possibly reduce the code bloat
    /// caused by the polymorphic DirectionProvider.
    /// (This is a premature optimization. Evil.)
    fn init_graph_from_modules(
        modules: &HashMap<CompactString, (SVerilogModule, ModuleMap)>,
        (top_name, top_m, top_mm): (
            &CompactString,
            &SVerilogModule,
            &ModuleMap,
        ),
        lib: &impl LeafPinProvider,
    ) -> Option<NetlistDB> {
        let (est_num_cells, est_num_logic_pins) = estimate_size(
            modules,
            &mut HashSet::new(),
            (top_name, top_m, top_mm),
            &mut HashMap::new(),
        )?;
        let est_num_cells = est_num_cells + 1; // top level

        let mut db = NetlistDB {
            name: top_name.clone(),
            num_cells: 1,
            num_pins: 0,
            num_logic_pins: 0,
            num_nets: 0,
            cellname2id: HashMap::with_capacity(est_num_cells),
            logicpinname2id: HashMap::with_capacity(est_num_logic_pins),
            pinname2id: HashMap::new(),
            netname2id: HashMap::new(),
            portname2pinid: HashMap::new(),
            celltypes: Vec::with_capacity(est_num_cells),
            cellnames: Vec::with_capacity(est_num_cells),
            logicpintypes: Vec::with_capacity(est_num_logic_pins),
            logicpinnames: Vec::with_capacity(est_num_logic_pins),
            keep_logic_pins: HashSet::new(),
            keep_nets: HashSet::new(),
            keep_cells: HashSet::new(),
            pinid2logicpinid: Vec::new(),
            netnames: Vec::new(),
            pinnames: Vec::new(),
            pin2cell: UVec::new(),
            pin2net: UVec::new(),
            cell2pin: Default::default(),
            net2pin: Default::default(),
            pindirect: UVec::new(),
            cell2noutputs: UVec::new(),
            net_zero: None,
            net_one: None,
        };

        db.cellname2id.insert(HierName::empty(), 0);
        db.celltypes.push(top_name.clone());
        db.cellnames.push(HierName::empty());

        let mut net_sets = DisjointSet::with_capacity(est_num_logic_pins);

        db.build_modules(
            modules,
            (top_name, top_m, top_mm),
            &mut net_sets,
            HierName::empty(),
            lib,
            false,
        )?;

        if db.num_logic_pins > est_num_logic_pins {
            tracing::warn!(
                "there turns out to be more \
                 logic pin (net) than expected -- typically because there \
                 are nets undefined but used. It is not suggested to do so \
                 because this will lead to long construction time."
            );
        }

        // set TopPort property for top module ports.
        for port in &top_m.ports {
            let name = match port {
                SVerilogPortDef::Basic(name) => name,
                SVerilogPortDef::Conn(name, _) => name,
            };
            let w = top_mm.port_widths.get(name).copied();
            for w in enum_in_width(w) {
                let id = db.try_find_logic_pin(&HierName::empty(), name, w)?;
                db.logicpintypes[id] = LogicPinType::TopPort;
            }
        }

        // create net maps.
        // first finalize the disjoint set and compute the sizes.
        let finalized_net_sets = net_sets.finalize(db.num_logic_pins)?;
        let logicpin2nets = finalized_net_sets.logicbin2nets;
        db.num_nets = finalized_net_sets.num_nets;
        db.net_zero = finalized_net_sets.net_zero;
        db.net_one = finalized_net_sets.net_one;

        db.keep_nets = db
            .keep_logic_pins
            .iter()
            .map(|&lp_id| logicpin2nets[lp_id])
            .collect();
        db.keep_logic_pins = HashSet::new();

        // finalize pin index and pin-net mapping.
        db.pinid2logicpinid = db
            .logicpintypes
            .iter()
            .enumerate()
            .filter_map(|(id, t)| match t.is_pin() {
                false => None,
                true => Some(id),
            })
            .collect::<Vec<usize>>();
        db.num_pins = db.pinid2logicpinid.len();

        let mut ret_pinname2id = None;
        let mut ret_pin2cell = None;
        let mut ret_cell2pin = None;
        let mut ret_pinnames = None;
        let mut ret_pin2net = None;
        let mut ret_net2pin = None;
        let mut ret_netname2id = None;
        let mut ret_netnames = None;
        let mut ret_portname2pinid = None;

        rayon::scope(|s| {
            // pinname2id -> pin2cell -> cell2pin
            s.spawn(|_| {
                let pinname2id = db
                    .pinid2logicpinid
                    .iter()
                    .enumerate()
                    .map(|(id, logic_id)| {
                        (db.logicpinnames[*logic_id].clone(), id)
                    })
                    .collect::<HashMap<_, _>>();

                // finalize pin to cell map
                let mut pin2cell =
                    UVec::new_filled(usize::MAX, db.num_pins, Device::CPU);
                let pinidcell = pinname2id
                    .par_iter()
                    .map(|((hier, _, _), id)| {
                        (*id, *db.cellname2id.get(hier).unwrap())
                    })
                    .collect::<Vec<_>>();
                for (id, cellid) in pinidcell {
                    pin2cell[id] = cellid;
                }
                debug_assert!(!pin2cell.contains(&usize::MAX));

                // construct cell CSR
                let cell2pin =
                    VecCSR::from(db.num_cells, db.num_pins, &pin2cell);

                ret_pinname2id = Some(pinname2id);
                ret_pin2cell = Some(pin2cell);
                ret_cell2pin = Some(cell2pin);
            });

            // pinnames
            s.spawn(|_| {
                ret_pinnames = Some(
                    db.pinid2logicpinid
                        .iter()
                        .map(|logic_id| db.logicpinnames[*logic_id].clone())
                        .collect::<Vec<_>>(),
                );
            });

            // pin2net -> net2pin
            s.spawn(|_| {
                let pin2net = db
                    .pinid2logicpinid
                    .iter()
                    .map(|logic_id| logicpin2nets[*logic_id])
                    .collect::<UVec<_>>();

                // construct net CSR
                let net2pin = VecCSR::from(db.num_nets, db.num_pins, &pin2net);
                ret_pin2net = Some(pin2net);
                ret_net2pin = Some(net2pin);
            });

            // netname2id, netnames
            s.spawn(|_| {
                let netname2id = db
                    .logicpinnames
                    .iter()
                    .enumerate()
                    .filter_map(|(id, name)| {
                        match db.logicpintypes[id].is_net() {
                            false => None,
                            true => Some((name.clone(), logicpin2nets[id])),
                        }
                    })
                    .collect::<HashMap<_, _>>();
                let max_net_id =
                    netname2id.values().copied().max().unwrap_or(0);
                let mut netnames = vec![
                    (
                        HierName::empty(),
                        CompactString::new_inline(""),
                        None
                    );
                    max_net_id + 1
                ];

                // find the best name for each net
                for (netname, id) in &netname2id {
                    let current_name = &mut netnames[*id];
                    let current_hier_depth = current_name.0.iter().count();
                    let netname_hier_depth = netname.0.iter().count();
                    if current_name.1.is_empty()
                        || netname_hier_depth < current_hier_depth
                        || (netname_hier_depth == current_hier_depth
                            && netname.1 < current_name.1)
                    {
                        *current_name = netname.clone();
                    }
                }
                ret_netname2id = Some(netname2id);
                ret_netnames = Some(netnames);
            });

            // portname2pinid
            s.spawn(|_| {
                let logicpinid2pinid = db
                    .pinid2logicpinid
                    .iter()
                    .enumerate()
                    .map(|(i, &lpi)| (lpi, i))
                    .collect::<HashMap<_, _>>();
                let mut portname2pinid = HashMap::new();
                for port in &top_m.ports {
                    let (name, expr) = match port {
                        SVerilogPortDef::Basic(name) => (
                            name,
                            Cow::Owned(Wirexpr::Basic(WirexprBasic::Full(
                                name.clone(),
                            ))),
                        ),
                        SVerilogPortDef::Conn(name, expr) => {
                            (name, Cow::Borrowed(expr))
                        }
                    };
                    let width = top_mm.port_widths.get(name).copied();
                    for (w, eb) in
                        enum_in_width(width).zip(top_mm.eval_expr(&expr))
                    {
                        let (pname, pidx) = match eb {
                            ExprBit::Const(_) => continue,
                            ExprBit::Var(pname, pidx) => (pname, pidx),
                        };
                        let id = db
                            .try_find_logic_pin(&HierName::empty(), name, w)
                            .unwrap();
                        let id = *logicpinid2pinid.get(&id).unwrap();
                        portname2pinid.insert((pname.clone(), pidx), id);
                    }
                }
                ret_portname2pinid = Some(portname2pinid);
            });
        });

        db.pinname2id = ret_pinname2id.unwrap();
        db.pin2cell = ret_pin2cell.unwrap();
        db.cell2pin = ret_cell2pin.unwrap();
        db.pinnames = ret_pinnames.unwrap();
        db.pin2net = ret_pin2net.unwrap();
        db.net2pin = ret_net2pin.unwrap();
        db.netname2id = ret_netname2id.unwrap();
        db.netnames = ret_netnames.unwrap();
        db.portname2pinid = ret_portname2pinid.unwrap();

        Some(db)
    }

    /// Building a netlist database STEP 2:
    /// Assign directions to netlist pins given the macro and the pin type.
    #[must_use]
    fn assign_direction(
        &mut self,
        (_top_name, top_m, top_mm): (
            &CompactString,
            &SVerilogModule,
            &ModuleMap,
        ),
        lib: &impl DirectionProvider,
    ) -> Option<()> {
        // query the provider for cell pins
        self.pindirect = self
            .pinid2logicpinid
            .iter()
            .enumerate()
            .map(|(i, logic_id)| match self.logicpintypes[*logic_id] {
                LogicPinType::TopPort => Direction::Unknown,
                LogicPinType::LeafCellPin => {
                    let macro_name = &self.celltypes[self.pin2cell[i]];
                    let pin_name = &self.pinnames[i];
                    lib.direction_of(macro_name, &pin_name.1, pin_name.2)
                }
                _ => unreachable!(),
            })
            .collect();

        // query the module definition for ports
        for port in &top_m.ports {
            use ExprBit::*;
            let (name, ref_names) = match port {
                SVerilogPortDef::Basic(name) => {
                    (name, Either::Left(std::iter::repeat(name)))
                }
                SVerilogPortDef::Conn(name, expr) => {
                    (
                        name,
                        Either::Right(top_mm.eval_expr(expr).map(
                            |eb| match eb {
                                Const(_) => {
                                    tracing::error!("Literal unsupported");
                                    panic!() // for simplicity here.
                                }
                                Var(pname, _) => pname,
                            },
                        )),
                    )
                }
            };
            let width = top_mm.port_widths.get(name).copied();
            for (id, ref_name) in enum_in_width(width).zip(ref_names) {
                let k = (HierName::empty(), name.clone(), id);
                let deftype = match top_mm.def_types.get(ref_name) {
                    Some(v) => v,
                    None => {
                        tracing::error!(
                            "io reference {}/{}{:?} not found,\
                             required for direction discovery.",
                            k.0,
                            k.1,
                            k.2
                        );
                        return None;
                    }
                };
                let dir = match deftype {
                    WireDefType::Input => {
                        // input port is net output
                        Direction::O
                    }
                    WireDefType::Output => Direction::I,
                    WireDefType::InOut => {
                        tracing::warn!(
                            "inout unsupported for pin {}/{}{:?}, treating as unknown.",
                            k.0,
                            k.1,
                            k.2
                        );
                        Direction::Unknown
                    }
                    WireDefType::Wire => {
                        tracing::error!(
                            "named port connection {name} should \
                             not refer to non-io wire {ref_name}."
                        );
                        return None;
                    }
                };
                self.pindirect[*self.pinname2id.get(&k).unwrap()] = dir;
            }
        }

        let num_unknowns = self
            .pindirect
            .iter()
            .filter(|t| **t == Direction::Unknown)
            .count();

        if num_unknowns != 0 && lib.should_warn_missing_directions() {
            tracing::warn!(
                "There are {num_unknowns} pins with unknown directions"
            );
        }

        self.post_assign_direction()?;
        Some(())
    }

    /// Ensure that the first pin of net2pin.item is the driver pin.
    #[must_use]
    fn post_assign_direction(&mut self) -> Option<()> {
        let net_outs: Vec<Vec<usize>> = (0..self.num_nets)
            .into_par_iter()
            .map(|i| {
                let l = self.net2pin.start[i];
                let r = self.net2pin.start[i + 1];
                (l..r)
                    .filter(|&j| {
                        self.pindirect[self.net2pin.items[j]] == Direction::O
                    })
                    .collect()
            })
            .collect();

        let mut num_undriven_nets = 0;
        self.cell2noutputs =
            UVec::new_filled(0usize, self.num_cells, Device::CPU);
        for (i, outs) in net_outs.iter().enumerate() {
            match outs.len() {
                0 => {
                    if Some(i) != self.net_zero && Some(i) != self.net_one {
                        num_undriven_nets += 1;
                    }
                }
                1 => {
                    let p = outs[0];
                    let driver_pin = self.net2pin.items[p];
                    self.cell2noutputs[self.pin2cell[driver_pin]] += 1;

                    // flip to guarantee that the net root is the first node in
                    // net CSR
                    let l = self.net2pin.start[i];
                    self.net2pin.items.swap(l, p);
                }
                _ => {
                    // TODO return an error
                    tracing::error!(
                        "There must be exactly one driver for each net. \
                     The net {i} has outputs {outs:?}"
                    );
                    return None;
                }
            }
        }

        if num_undriven_nets != 0 {
            tracing::warn!(
                "There are {num_undriven_nets} nets without driving pins."
            );
        }

        // For each kept cell, mark the nets it drives as keep so the
        // optimizer cannot eliminate the driver connection.
        let kept_output_nets: Vec<usize> = self
            .keep_cells
            .iter()
            .flat_map(|&cell_id| {
                let start = self.cell2pin.start[cell_id];
                let end = self.cell2pin.start[cell_id + 1];
                (start..end).filter_map(|j| {
                    let pin_id = self.cell2pin.items[j];
                    (self.pindirect[pin_id] == Direction::O)
                        .then(|| self.pin2net[pin_id])
                })
            })
            .collect();
        self.keep_nets.extend(kept_output_nets);

        Some(())
    }
}
