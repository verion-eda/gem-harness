# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = [
#   "pyslang",
# ]
# ///
"""
Test that dpi_preprocess.py correctly handles DPI output parameters
with reset-path blocking assignments (the dual-driver bug).

Run:  uv run --script examples/dpi/scripts/test_dpi_output_reset.py
"""

from __future__ import annotations

import os
import re
import sys

# Add the scripts directory to sys.path so we can import the preprocessor.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import dpi_preprocess as dpp  # noqa: E402


def _run(source: str) -> str:
    return dpp.preprocess(source)


def _run_file(rel_path: str) -> str:
    path = os.path.join(os.path.dirname(SCRIPT_DIR), rel_path)
    with open(path) as f:
        source = f.read()
    return _run(source)


def test_output_reset_sv() -> None:
    """Verify dpi_output_reset.sv: output vars become wire, no blocking assigns,
    cell connections correct, non-blocking register assigns preserved."""
    result = _run_file("rtl/dpi_output_reset.sv")

    # Output vars should be declared as wire, not bit/int/reg
    assert "wire __ready;" in result, (
        f"Expected 'wire __ready;' declaration.\nGot:\n{result}"
    )
    assert "wire [31:0] __data;" in result, (
        f"Expected 'wire [31:0] __data;' declaration.\nGot:\n{result}"
    )

    # No blocking assignments to DPI output vars should remain
    assert not re.search(r"^\s*__ready\s*=[^;]+;", result, re.MULTILINE), (
        "Blocking assignment to __ready should be removed"
    )
    assert not re.search(r"^\s*__data\s*=[^;]+;", result, re.MULTILINE), (
        "Blocking assignment to __data should be removed"
    )

    # Cell connections should be present
    assert ".OUT_0(__ready)" in result, "Expected .OUT_0(__ready) connection"
    assert ".OUT_1(__data)" in result, "Expected .OUT_1(__data) connection"
    assert ".IN_0(ctrl)" in result, "Expected .IN_0(ctrl) connection"

    # Non-blocking register assigns should be preserved
    assert re.search(r"__ready_reg\s*<=", result), (
        "Non-blocking assign to __ready_reg should be preserved"
    )
    assert re.search(r"__data_reg\s*<=", result), (
        "Non-blocking assign to __data_reg should be preserved"
    )

    print("PASS: test_output_reset_sv")


def test_simdram_no_dual_driver() -> None:
    """Verify SimDRAM.v: no blocking assigns to DPI output vars, all declared as wire."""
    sv_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), "..", "gemmini", "generated-rtl", "SimDRAM.v"
    )
    if not os.path.exists(sv_path):
        print("SKIP: test_simdram_no_dual_driver (SimDRAM.v not found)")
        return

    with open(sv_path) as f:
        source = f.read()
    result = _run(source)

    dpi_out_vars = [
        "__ar_ready",
        "__aw_ready",
        "__w_ready",
        "__r_valid",
        "__r_last",
        "__b_valid",
    ]
    for var in dpi_out_vars:
        # No blocking assignments
        assert not re.search(
            rf"^\s*\b{re.escape(var)}\b\s*=[^;]+;", result, re.MULTILINE
        ), f"Blocking assignment to {var} should be removed"

        # Should be declared as wire
        assert re.search(
            rf"^\s*wire\b.*\b{re.escape(var)}\b\s*;", result, re.MULTILINE
        ), f"{var} should be declared as wire"

    print("PASS: test_simdram_no_dual_driver")


def test_simuart_no_dual_driver() -> None:
    """Verify SimUART.v: no blocking assigns to DPI output vars."""
    sv_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), "..", "gemmini", "generated-rtl", "SimUART.v"
    )
    if not os.path.exists(sv_path):
        print("SKIP: test_simuart_no_dual_driver (SimUART.v not found)")
        return

    with open(sv_path) as f:
        source = f.read()
    result = _run(source)

    # Collect actual DPI output vars from the result
    dpi_out_vars: set[str] = set()
    for m in re.finditer(r"\$__DPI__\w+[^;]*?\.OUT_\d+\((\w+)\)", result):
        dpi_out_vars.add(m.group(1))

    for var in dpi_out_vars:
        assert not re.search(
            rf"^\s*\b{re.escape(var)}\b\s*=[^;]+;", result, re.MULTILINE
        ), f"Blocking assignment to {var} should be removed"

        assert re.search(
            rf"^\s*wire\b.*\b{re.escape(var)}\b\s*;", result, re.MULTILINE
        ), f"{var} should be declared as wire"

    print("PASS: test_simuart_no_dual_driver")


def test_existing_output_param_unchanged() -> None:
    """Regression: dpi_output_param.sv still works (output becomes wire [31:0])."""
    result = _run_file("rtl/dpi_output_param.sv")

    # The output param variable should become a wire
    assert "wire [31:0] dpi_data;" in result, (
        f"Expected 'wire [31:0] dpi_data;' declaration.\nGot:\n{result}"
    )

    # Cell connection should still be present
    assert ".OUT_0(dpi_data)" in result, "Expected .OUT_0(dpi_data) connection"

    # The blackbox should still be generated
    assert "$__DPI__get_value" in result, "Expected $__DPI__get_value blackbox"

    print("PASS: test_existing_output_param_unchanged")


if __name__ == "__main__":
    test_output_reset_sv()
    test_simdram_no_dual_driver()
    test_simuart_no_dual_driver()
    test_existing_output_param_unchanged()
    print("\nAll tests passed!")
