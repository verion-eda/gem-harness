// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0

use simulator::{InteractiveSimulator, SimulatorBuilder, Offset};

pub struct Alu {
    pub sim: InteractiveSimulator,

    // Inputs
    pub a: u64,
    pub b: u64,
    pub op: u64,

    // Outputs
    pub carry: bool,
    pub overflow: bool,
    pub result: u64,
    pub sign: bool,
    pub zero: bool,

    handles: AluHandles,
}

struct AluHandles {
    a: Vec<Offset>,
    b: Vec<Offset>,
    carry: Vec<Offset>,
    op: Vec<Offset>,
    overflow: Vec<Offset>,
    result: Vec<Offset>,
    sign: Vec<Offset>,
    zero: Vec<Offset>,

}

impl Alu {
    pub fn new(builder: SimulatorBuilder) -> anyhow::Result<Self> {
        let sim = builder.build()?;

        let handles = AluHandles {
            a: resolve_offsets(&sim, "a")?,
            b: resolve_offsets(&sim, "b")?,
            carry: resolve_offsets(&sim, "carry")?,
            op: resolve_offsets(&sim, "op")?,
            overflow: resolve_offsets(&sim, "overflow")?,
            result: resolve_offsets(&sim, "result")?,
            sign: resolve_offsets(&sim, "sign")?,
            zero: resolve_offsets(&sim, "zero")?,
        };

        Ok(Self {
            sim,
            a: 0,
            b: 0,
            op: 0,
            carry: false,
            overflow: false,
            result: 0,
            sign: false,
            zero: false,
            handles,
        })
    }

    fn flush_inputs(&self, sim: &mut InteractiveSimulator) {
        sim.poke_offsets(&self.handles.a, self.a);
        sim.poke_offsets(&self.handles.b, self.b);
        sim.poke_offsets(&self.handles.op, self.op);
    }

    fn fetch_outputs(&mut self, sim: &InteractiveSimulator) {
        self.carry = sim.peek_offsets(&self.handles.carry) != 0;
        self.overflow = sim.peek_offsets(&self.handles.overflow) != 0;
        self.result = sim.peek_offsets(&self.handles.result);
        self.sign = sim.peek_offsets(&self.handles.sign) != 0;
        self.zero = sim.peek_offsets(&self.handles.zero) != 0;
    }
}

fn alu_flush_inputs_impl(handles: &AluHandles, a: u64, b: u64, op: u64, sim: &mut InteractiveSimulator) {
    sim.poke_offsets(&handles.a, a);
    sim.poke_offsets(&handles.b, b);
    sim.poke_offsets(&handles.op, op);
}

impl Alu {
    pub fn eval(&mut self) {
        alu_flush_inputs_impl(&self.handles, self.a, self.b, self.op, &mut self.sim);
        self.sim.step();
        self.sim.sync();
        self.carry = self.sim.peek_offsets(&self.handles.carry) != 0;
        self.overflow = self.sim.peek_offsets(&self.handles.overflow) != 0;
        self.result = self.sim.peek_offsets(&self.handles.result);
        self.sign = self.sim.peek_offsets(&self.handles.sign) != 0;
        self.zero = self.sim.peek_offsets(&self.handles.zero) != 0;
    }
}

fn resolve_offsets(sim: &InteractiveSimulator, name: &str) -> anyhow::Result<Vec<Offset>> {
    sim.resolve_signal_offsets(name)
        .ok_or_else(|| anyhow::anyhow!("Signal {} not found", name))
}

