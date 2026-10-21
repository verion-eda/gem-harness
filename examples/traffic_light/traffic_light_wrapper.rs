// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use simulator::{
    InteractiveSimulator, SimulatorBuilder, StateOffset, signals::SignalPath,
};

pub struct TrafficLightWrapper {
    pub sim: InteractiveSimulator,

    // Inputs
    pub clk: u64,
    pub rst: u64,

    // Outputs
    pub green: bool,
    pub red: bool,
    pub yellow: bool,

    handles: TrafficLightWrapperHandles,
}

struct TrafficLightWrapperHandles {
    clk: Vec<StateOffset>,
    green: Vec<StateOffset>,
    red: Vec<StateOffset>,
    rst: Vec<StateOffset>,
    yellow: Vec<StateOffset>,
}

impl TrafficLightWrapper {
    pub fn new(builder: SimulatorBuilder) -> anyhow::Result<Self> {
        let sim = builder.build()?;

        let handles = TrafficLightWrapperHandles {
            clk: resolve_offsets(&sim, "clk")?,
            green: resolve_offsets(&sim, "green")?,
            red: resolve_offsets(&sim, "red")?,
            rst: resolve_offsets(&sim, "rst")?,
            yellow: resolve_offsets(&sim, "yellow")?,
        };

        Ok(Self {
            sim,
            clk: 0,
            rst: 0,
            green: false,
            red: false,
            yellow: false,
            handles,
        })
    }

    fn flush_inputs(&mut self) {
        self.sim.poke_offsets(&self.handles.clk, self.clk);
        self.sim.poke_offsets(&self.handles.rst, self.rst);
    }

    fn fetch_outputs(&mut self) {
        self.green = self.sim.peek_offsets(&self.handles.green) != 0;
        self.red = self.sim.peek_offsets(&self.handles.red) != 0;
        self.yellow = self.sim.peek_offsets(&self.handles.yellow) != 0;
    }

    pub fn eval(&mut self) {
        self.flush_inputs();
        self.sim.step();
        self.sim.sync();
        self.fetch_outputs();
    }
}

fn resolve_offsets(
    sim: &InteractiveSimulator,
    name: &str,
) -> anyhow::Result<Vec<StateOffset>> {
    let sp: SignalPath = name.parse()?;
    sim.resolve_signal_offsets(&sp).ok_or_else(|| {
        anyhow::anyhow!("failed to get offset for signal {name}")
    })
}
