# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
"""
gem-harness - GPU-accelerated hardware simulator

This module provides Python bindings for the hardware simulator,
which simulates digital circuits represented as And-Inverter Graphs (AIGs)
on CUDA GPUs.
"""

from typing import Any, List

class Simulator:
    """
    Interactive hardware simulator for AIG netlists

    This class provides a Python interface to simulate digital circuits
    represented as And-Inverter Graphs (AIGs). It runs on CUDA GPUs for
    high-performance simulation.

    Args:
        netlist_path: Path to the Verilog netlist file
        gemparts_path: Path to the GEM partitions file
        num_blocks: Number of CUDA blocks to use for simulation
        device_id: CUDA device ID (0 for first GPU, 1 for second, etc.)

    Example:
        >>> sim = Simulator("netlist.v", "gemparts.txt", num_blocks=256, device_id=0)
        >>> sim.reset()
        >>> sim.poke_signal("clk", True)
        >>> sim.step()
        >>> result = sim.peek_signal("output")
    """

    def __init__(
        self,
        netlist_path: str,
        gemparts_path: str,
        num_blocks: int,
        device_id: int,
    ) -> None:
        """
        Create a new simulator instance

        Args:
            netlist_path: Path to the Verilog netlist file
            gemparts_path: Path to the GEM partitions file
            num_blocks: Number of CUDA blocks to use for simulation
            device_id: CUDA device ID (0 for first GPU, 1 for second, etc.)

        Raises:
            RuntimeError: If simulator creation fails
        """
        ...

    def reset(self) -> None:
        """
        Reset the simulator to initial state

        This must be called once after creating the simulator and before
        any poke/step/peek operations.

        Example:
            >>> sim.reset()
        """
        ...

    def poke(self, aigpin: int, value: bool) -> None:
        """
        Set an input signal value by AIG pin number

        Args:
            aigpin: AIG pin number (without invert bit)
            value: Signal value (True for high, False for low)

        Raises:
            RuntimeError: If reset() hasn't been called yet

        Example:
            >>> sim.poke(42, True)
        """
        ...

    def poke_signal(self, signal_path: str, value: bool) -> None:
        """
        Set an input signal value by signal name

        Args:
            signal_path: Signal name (e.g., "clk" or "cpu/alu/in[0]")
            value: Signal value (True for high, False for low)

        Raises:
            RuntimeError: If reset() hasn't been called or signal not found

        Example:
            >>> sim.poke_signal("cpu/clk", True)
            >>> sim.poke_signal("data[7:0]", False)
        """
        ...

    def poke_bits(self, aigpins: List[int], value: int) -> None:
        """
        Set multiple input signal values at once

        This is more efficient than calling poke() multiple times.
        The value is interpreted as a bit vector where bit i corresponds
        to aigpins[i].

        Args:
            aigpins: List of AIG pin numbers (without invert bit)
            value: 64-bit value where bit i corresponds to aigpins[i]

        Raises:
            RuntimeError: If reset() hasn't been called or pins not found
            ValueError: If more than 64 pins are provided

        Example:
            >>> # Set pins 10,11,12,13 to binary 1010
            >>> sim.poke_bits([10, 11, 12, 13], 0b1010)
        """
        ...

    def step(self) -> None:
        """
        Execute one simulation cycle

        This propagates the input values through the circuit logic.
        Call poke/poke_signal before step, then peek/peek_signal after.

        Example:
            >>> sim.poke_signal("clk", True)
            >>> sim.step()
            >>> result = sim.peek_signal("output")
        """
        ...

    def peek(self, aigpin: int) -> bool:
        """
        Read an output signal value by AIG pin number

        Args:
            aigpin: AIG pin number (with invert bit in LSB)

        Returns:
            Signal value (True for high, False for low)

        Example:
            >>> value = sim.peek(100)
        """
        ...

    def peek_signal(self, signal_path: str) -> int:
        """
        Read an output signal value by signal name

        Args:
            signal_path: Signal name (e.g., "output" or "cpu/alu/result[0]")

        Returns:
            Signal value as a 64-bit unsigned integer

        Raises:
            RuntimeError: If signal not found

        Example:
            >>> value = sim.peek_signal("cpu/output[7:0]")
            >>> print(f"Output: 0x{value:02x}")
        """
        ...

    def peek_bits(self, aigpins: List[int]) -> int:
        """
        Read multiple output signal values at once

        This is more efficient than calling peek() multiple times.
        Returns a 64-bit value where bit i corresponds to aigpins[i].

        Args:
            aigpins: List of AIG pin numbers (with invert bit in LSB)

        Returns:
            64-bit value where bit i corresponds to aigpins[i]

        Raises:
            RuntimeError: If pins not found
            ValueError: If more than 64 pins are provided

        Example:
            >>> # Read pins 20,21,22,23
            >>> value = sim.peek_bits([20, 21, 22, 23])
            >>> print(f"Output: {value:04b}")
        """
        ...

    def sync(self) -> None:
        """
        Synchronize GPU operations (wait for completion)

        This ensures all GPU operations have completed before proceeding.
        Useful for timing measurements or before accessing results.

        Example:
            >>> sim.step()
            >>> sim.sync()  # Wait for GPU to finish
        """
        ...

    def num_inputs(self) -> int:
        """
        Get the number of input pins in the circuit

        Returns:
            Number of input pins

        Example:
            >>> num_inputs = sim.num_inputs()
            >>> print(f"Circuit has {num_inputs} inputs")
        """
        ...

    def num_outputs(self) -> int:
        """
        Get the number of output pins in the circuit

        Returns:
            Number of output pins

        Example:
            >>> num_outputs = sim.num_outputs()
            >>> print(f"Circuit has {num_outputs} outputs")
        """
        ...

    def __enter__(self) -> "Simulator":
        """Context manager entry"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool:
        """Context manager exit - ensures GPU sync on exit"""
        ...

    def __repr__(self) -> str:
        """String representation of the simulator"""
        ...
