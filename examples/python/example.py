#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
"""
Example of using the gem-harness Python bindings

This demonstrates how to use the PyO3-based Python interface to the
hardware simulator using the adder example.
"""

from harness import Simulator


def main():
    # Use the adder example files (assumes they're already compiled)
    netlist_path = "examples/adder/build/gatelevel.gv"
    gemparts_path = "examples/adder/build/result.gemparts"

    sim = Simulator(
        netlist_path=netlist_path,
        gemparts_path=gemparts_path,
        num_blocks=1,
        device_id=0,
    )

    print(sim)
    print(f"Inputs: {sim.num_inputs()}, Outputs: {sim.num_outputs()}")

    sim.reset()

    sim.poke_signal("a", True)
    sim.poke_signal("b", False)
    sim.step()
    res_sum = sim.peek_signal("sum")
    print(f"sum: 0x{res_sum:02x}")
    carry = sim.peek_signal("carry")
    print(f"carry: 0x{carry:02x}")
    sim.sync()

    # Using context manager (automatic sync)
    with Simulator(netlist_path, gemparts_path, 1, 0) as sim2:
        sim2.reset()
        sim2.poke_signal("a", True)
        sim2.poke_signal("b", True)
        sim2.step()
        res_sum = sim2.peek_signal("sum")
        print(f"sum: 0x{res_sum:02x}")
        carry = sim2.peek_signal("carry")
        print(f"carry: 0x{carry:02x}")
    # sim2.sync() is called automatically on exit


if __name__ == "__main__":
    main()
