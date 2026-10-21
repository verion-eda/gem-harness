# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = [
#   "pyslang",
# ]
# ///
"""
Test that dpi_preprocess.py correctly handles unpacked array DPI parameters.

Run:  uv run --script examples/dpi/scripts/test_dpi_array.py
"""

from __future__ import annotations

import os
import sys
import textwrap

# Add the scripts directory to sys.path so we can import the preprocessor.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import dpi_preprocess as dpp  # noqa: E402


def _run(source: str) -> str:
    return dpp.preprocess(source)


def _run_file(rel_path: str) -> str:
    path = os.path.join(os.path.dirname(SCRIPT_DIR), rel_path)
    with open(path) as f:
        return _run(f.read())


def test_array_detection() -> None:
    """Verify that unpacked array [] is detected on DPI ports."""
    import pyslang

    src = textwrap.dedent("""\
        import "DPI-C" function void f(
            input int scalar,
            input byte arr_in[],
            output byte arr_out[]
        );
        module m(); endmodule
    """)
    tree = pyslang.SyntaxTree.fromText(src)
    funcs, _ = dpp.parse_dpi_imports(tree, src)
    info = funcs["f"]
    # arg 0 (scalar) is not an array; args 1 and 2 are.
    assert info.unpacked_args_width == {1: 8, 2: 8}, f"Got: {info.unpacked_args_width}"
    print("PASS: test_array_detection")


def test_param_parsing() -> None:
    """Verify module parameter defaults are parsed."""
    src = textwrap.dedent("""\
        module m #(
            parameter ADDR_BITS=32, DATA_BITS = 64, ID_BITS = 5,
                      WORD_SIZE = DATA_BITS/8
        )(input clk);
        endmodule
    """)
    params = dpp._parse_module_params(src)
    assert params["ADDR_BITS"] == 32
    assert params["DATA_BITS"] == 64
    assert params["WORD_SIZE"] == 8, f"Got: {params.get('WORD_SIZE')}"
    print("PASS: test_param_parsing")


def test_var_decl_lookup() -> None:
    """Verify variable declaration parsing for unpacked arrays."""
    src = "    byte __w_data[(DATA_BITS / 8)-1:0];\n"
    result = dpp._find_array_var_decl(src, "__w_data")
    assert result is not None
    elem_w, size_expr = result
    assert elem_w == 8
    assert size_expr == "(DATA_BITS / 8)", f"Got: {size_expr!r}"
    print("PASS: test_var_decl_lookup")


def test_total_width_computation() -> None:
    """Verify total width = element_width × array_size."""
    params = {"DATA_BITS": 64}
    total = dpp._compute_total_width(8, "(DATA_BITS / 8)", params)
    assert total == 64, f"Got: {total}"
    print("PASS: test_total_width_computation")


def test_dpi_array_sv() -> None:
    """Full integration test on dpi_array.sv."""
    result = _run_file("rtl/dpi_array.sv")

    # The cell name should encode 64-bit widths for the array ports,
    # not 8-bit (the byte element width).
    assert "$I32_64$O64" in result, (
        f"Expected cell name with 64-bit array widths.\n"
        f"Cell names found: {[line for line in result.splitlines() if '$__DPI__' in line]}"
    )

    # Unpacked array declarations should be replaced with packed vectors.
    assert "logic [63:0] __w_data;" in result, (
        "Expected packed logic [63:0] __w_data declaration"
    )
    # __r_data is a DPI output param, so it becomes wire after fixup.
    assert "wire [63:0] __r_data;" in result, (
        "Expected wire [63:0] __r_data declaration (DPI output)"
    )

    # Byte-level index access should be rewritten to part-select.
    # At least one of the generate-block accesses should be fixed up.
    assert "__w_data[(i)*8 +: 8]" in result or "__w_data[(i) * 8 +: 8]" in result, (
        "Expected __w_data index fixup to part-select"
    )
    assert "__r_data[(i)*8 +: 8]" in result or "__r_data[(i) * 8 +: 8]" in result, (
        "Expected __r_data index fixup to part-select"
    )

    # The blackbox port widths should be 64, not 8.
    assert "[63:0] IN_1" in result, "Expected 64-bit IN_1 (w_data)"
    assert "[63:0] OUT_0" in result, "Expected 64-bit OUT_0 (r_data)"

    # Original 8-bit-only port should NOT appear for array ports.
    # (IN_0 for scalar 'ctrl' is 32-bit, that's fine.)
    assert "[7:0] IN_1" not in result, "8-bit IN_1 should not appear"

    print("PASS: test_dpi_array_sv")


def test_simdram() -> None:
    """Integration test on the actual SimDRAM.v from Gemmini."""
    sv_path = os.path.join(
        os.path.dirname(SCRIPT_DIR),
        "..",
        "gemmini",
        "generated-rtl",
        "SimDRAM.v",
    )
    try:
        with open(sv_path) as f:
            source = f.read()
    except FileNotFoundError:
        print("SKIP: test_simdram (SimDRAM.v not found)")
        return

    result = _run(source)

    cell_lines = [line for line in result.splitlines() if "$__DPI__memory_tick" in line]
    assert cell_lines, "Expected $__DPI__memory_tick cell"
    # Extract the input widths section (between $I and $O) and verify
    # array ports use 64-bit widths, not the 8-bit element width.
    cell_name = cell_lines[0].split()[0]
    input_widths = cell_name.split("$I")[1].split("$O")[0]
    widths = [int(w) for w in input_widths.split("_") if w]
    assert 64 in widths, (
        f"Expected 64-bit array port width in cell name, got widths: {widths}"
    )

    assert "logic [63:0] __w_data;" in result, (
        "Expected packed logic [63:0] __w_data declaration"
    )
    assert "wire [63:0] __r_data;" in result, (
        "Expected wire [63:0] __r_data declaration (DPI output)"
    )

    print("PASS: test_simdram")


def test_no_array_unchanged() -> None:
    """Non-array DPI functions should be unaffected."""
    result = _run_file("rtl/dpi_counter.sv")

    # Basic sanity: should still have the blackbox modules
    assert "$__DPI__sat_add" in result
    assert "$__DPI__trace_event" in result
    print("PASS: test_no_array_unchanged")


if __name__ == "__main__":
    test_array_detection()
    test_param_parsing()
    test_var_decl_lookup()
    test_total_width_computation()
    test_dpi_array_sv()
    test_simdram()
    test_no_array_unchanged()
    print("\nAll tests passed!")
