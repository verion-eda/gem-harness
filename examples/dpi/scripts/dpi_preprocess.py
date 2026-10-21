# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = [
#   "pyslang",
# ]
# ///
"""
Preprocess SystemVerilog files to replace DPI-C function imports and calls
with synthesizable blackbox module instantiations ($__DPI__<name>).

Uses pyslang to parse the full AST, so it handles DPI calls in any
expression context (ternaries, nested expressions, assign, always, etc.).

Usage:
    python dpi_preprocess.py input.sv [-o output.sv]

If -o is not specified, writes to stdout.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Any, cast

import pyslang


@dataclass
class DpiFuncInfo:
    """Parsed information about a single DPI-C function import."""

    args: list[tuple[str, int, str]] = field(default_factory=list)
    input_args: list[tuple[str, int]] = field(default_factory=list)
    output_args: list[tuple[str, int]] = field(default_factory=list)
    return_width: int | None = None
    # Maps arg index → element bit-width for params with unpacked array [].
    unpacked_args_width: dict[int, int] = field(default_factory=dict)


@dataclass
class DpiCall:
    """A single DPI function call site found in the source."""

    func_name: str
    arg_texts: list[str]
    start: int
    end: int


# ---------------------------------------------------------------------------
# Type width helpers
# ---------------------------------------------------------------------------

_TYPE_WIDTH: dict[str, int] = {
    "int": 32,
    "integer": 32,
    "shortint": 16,
    "longint": 64,
    "byte": 8,
    "bit": 1,
    "logic": 1,
    "reg": 1,
    "real": 64,
    "shortreal": 32,
    "chandle": 64,
    "string": 64,
}

# Alternation pattern matching any _TYPE_WIDTH keyword (used in regexes).
_TYPE_KW_PAT: str = "|".join(re.escape(k) for k in _TYPE_WIDTH)


def _safe_eval_int(expr: str, params: dict[str, int]) -> int:
    """Evaluate a simple arithmetic expression using *params* as variables.

    Only integer arithmetic is supported; builtins are disabled.
    """
    return int(eval(expr, {"__builtins__": {}}, params))  # noqa: S307


def _resolve_width(type_node: pyslang.SyntaxNode | None) -> int:
    """Extract bit width from a pyslang type syntax node.

    Handles keyword types (int, shortint, longint, byte, bit, logic, real,
    shortreal, chandle, string), explicit packed dimensions like
    ``logic [N:0]``, and falls back to 32.
    """
    if type_node is None:
        return 0
    # Check for packed dimensions (e.g. logic [15:0])
    if hasattr(type_node, "dimensions") and type_node.dimensions is not None:
        for dim in type_node.dimensions:
            if "RangeDimensionSpecifier" in str(dim.kind) or "Range" in str(dim.kind):
                # Try to extract the range selector
                sel = dim.selector if hasattr(dim, "selector") else dim
                if hasattr(sel, "left") and hasattr(sel, "right"):
                    try:
                        left = int(
                            str(sel.left.literal.valueText)
                            if hasattr(sel.left, "literal")
                            else str(sel.left.valueText)
                        )
                        right = int(
                            str(sel.right.literal.valueText)
                            if hasattr(sel.right, "literal")
                            else str(sel.right.valueText)
                        )
                        return abs(left - right) + 1
                    except (ValueError, AttributeError):
                        pass
    if hasattr(type_node, "keyword"):
        return _TYPE_WIDTH.get(str(type_node.keyword.valueText), 32)
    # Fallback
    return 32


# ---------------------------------------------------------------------------
# Unpacked array resolution helpers
# ---------------------------------------------------------------------------


def _parse_module_params(source: str) -> dict[str, int]:
    """Extract parameter defaults from the module header as name→int.

    Only captures simple integer literals and single-term expressions that
    can be evaluated from previously-defined parameters (e.g. ``DATA_BITS/8``).
    """
    params: dict[str, int] = {}
    # Match the parameter list between '#(' and the closing ')' of the header.
    hdr = re.search(r"#\s*\((.*?)\)\s*\(", source, re.DOTALL)
    if not hdr:
        return params
    param_text = hdr.group(1)
    # Split on commas not inside parens
    for item in re.split(r",(?![^()]*\))", param_text):
        m = re.search(r"(\w+)\s*=\s*(.+?)(?:,|$)", item.strip())
        if not m:
            continue
        name = m.group(1)
        expr = m.group(2).strip().rstrip(",")
        try:
            params[name] = _safe_eval_int(expr, params)
        except (ValueError, SyntaxError, KeyError, ZeroDivisionError, TypeError):
            pass  # expression depends on unresolved params or is not arithmetic
    return params


def _find_array_var_decl(source: str, var_name: str) -> tuple[int, str] | None:
    """Find a variable declaration with unpacked array bounds in the source.

    Returns ``(element_bit_width, array_size_expression)`` or ``None``.

    For ``byte __w_data[(DATA_BITS / 8)-1:0];`` returns ``(8, "(DATA_BITS / 8)")``.
    """
    # Match: type_keyword  var_name  [high_expr : low_expr]
    pattern = (
        rf"\b({_TYPE_KW_PAT})\s+"
        rf"{re.escape(var_name)}\s*"
        r"\[([^\]]+)\]\s*;"
    )
    m = re.search(pattern, source)
    if not m:
        return None
    elem_type = m.group(1)
    elem_w = _TYPE_WIDTH.get(elem_type, 8)
    bounds = m.group(2).strip()

    # Parse [high:low] — extract array size expression.
    if ":" in bounds:
        high_expr, low_expr = bounds.rsplit(":", 1)
        high_expr = high_expr.strip()
        low_expr = low_expr.strip()
        # Common pattern: [expr-1 : 0] → size = expr
        m2 = re.match(r"^(.+?)\s*-\s*1$", high_expr)
        if m2 and low_expr == "0":
            return (elem_w, m2.group(1).strip())
        # General case: size = high - low + 1
        return (elem_w, f"({high_expr}) - ({low_expr}) + 1")
    else:
        # [N] form — size is N
        return (elem_w, bounds.strip())


def _compute_total_width(elem_w: int, size_expr: str, params: dict[str, int]) -> int:
    """Compute total packed bit-width: element_width × array_size.

    Tries numeric evaluation using *params*; returns an ``int`` on success.
    """
    try:
        size = _safe_eval_int(size_expr, params)
        return elem_w * size
    except Exception:
        raise ValueError(
            f"Cannot resolve array size expression '{size_expr}' "
            f"with parameters {params}"
        )


def _resolve_array_params(
    source: str,
    dpi_funcs: dict[str, DpiFuncInfo],
    calls: list[DpiCall],
) -> dict[str, tuple[int, int]]:
    """Cross-reference DPI call args with module body to resolve array widths.

    Updates ``DpiFuncInfo`` widths in-place for array parameters.

    Returns ``{var_name: (element_bit_width, total_bit_width)}`` for each
    variable that was identified as an unpacked array fed to a DPI call.
    """
    params = _parse_module_params(source)
    converted: dict[str, tuple[int, int]] = {}

    for call in calls:
        info = dpi_funcs[call.func_name]
        if not info.unpacked_args_width:
            continue
        for arg_idx, elem_w in info.unpacked_args_width.items():
            if arg_idx >= len(call.arg_texts):
                continue
            var_name = call.arg_texts[arg_idx].strip()
            decl = _find_array_var_decl(source, var_name)
            if decl is None:
                continue
            _decl_elem_w, size_expr = decl
            total_w = _compute_total_width(elem_w, size_expr, params)
            converted[var_name] = (elem_w, total_w)

            # Update args
            pname, _old_w, direction = info.args[arg_idx]
            info.args[arg_idx] = (pname, total_w, direction)

            # Update input_args / output_args
            if direction == "output":
                out_idx = sum(1 for _, _, d in info.args[:arg_idx] if d == "output")
                info.output_args[out_idx] = (pname, total_w)
            else:
                in_idx = sum(1 for _, _, d in info.args[:arg_idx] if d != "output")
                info.input_args[in_idx] = (pname, total_w)

    return converted


def _fixup_unpacked_arrays(source: str, converted: dict[str, tuple[int, int]]) -> str:
    """Replace unpacked array declarations with packed logic vectors.

    For each variable in *converted*:
    - ``byte var[bounds]`` → ``logic [total_w-1:0] var``
    - ``var[idx]`` → ``var[(idx)*elem_w +: elem_w]``  (index access fixup)
    """
    if not converted:
        return source

    for var_name, (elem_w, total_w) in converted.items():
        # Replace declaration: type var[...]; → logic [total_w-1:0] var;
        decl_pat = (
            rf"\b({_TYPE_KW_PAT})\s+"
            rf"{re.escape(var_name)}\s*"
            r"\[[^\]]+\]\s*;"
        )
        range_str = f"[{total_w - 1}:0]" if total_w > 1 else ""
        source = re.sub(
            decl_pat,
            f"logic {range_str} {var_name};",
            source,
        )

        # Fix indexed accesses: var[expr] → var[(expr)*elem_w +: elem_w]
        # Only match simple index access (no ':' or '+:' inside brackets).
        idx_pat = rf"\b{re.escape(var_name)}\[([^\]:]+?)\]"

        # Bind var_name/elem_w via defaults to avoid late-binding closure issue.
        def _repl_idx(m: re.Match[str], _vn: str = var_name, _ew: int = elem_w) -> str:
            idx_expr = m.group(1).strip()
            return f"{_vn}[({idx_expr})*{_ew} +: {_ew}]"

        source = re.sub(idx_pat, _repl_idx, source)

    return source


# ---------------------------------------------------------------------------
# Parse DPI-C imports from the syntax tree
# ---------------------------------------------------------------------------


def parse_dpi_imports(
    tree: pyslang.SyntaxTree, source: str
) -> tuple[dict[str, DpiFuncInfo], list[tuple[int, int]]]:
    """
    Walk the syntax tree and extract DPI-C import declarations.
    Returns: (dpi_funcs, import_ranges) where
      dpi_funcs = {func_name: {args, input_args, output_args, ret_width}}
      import_ranges = [(start_offset, end_offset), ...] for each import statement
    """
    dpi_funcs: dict[str, DpiFuncInfo] = {}
    import_ranges: list[tuple[int, int]] = []
    root = tree.root
    nodes = cast(list[Any], root.members if hasattr(root, "members") else [root])

    for node in nodes:
        if "DPIImport" not in str(node.kind):
            continue
        method = node.method
        if method is None:
            continue

        # Record the source range of the entire import statement so we can
        # remove it precisely (no fragile regex needed).
        sr = node.sourceRange
        start = sr.start.offset
        end = sr.end.offset
        # Consume any trailing newline so we don't leave blank lines.
        if end < len(source) and source[end] == "\n":
            end += 1
        import_ranges.append((start, end))

        func_name = str(
            method.name.identifier.valueText
            if hasattr(method.name, "identifier")
            else method.name.valueText
        )

        # Return type
        ret_width: int | None = None
        if method.returnType is not None:
            ret_str = (
                str(method.returnType.keyword.valueText)
                if hasattr(method.returnType, "keyword")
                else ""
            )
            if ret_str and ret_str != "void":
                ret_width = _resolve_width(method.returnType)

        # Ports — now track direction per port
        args: list[tuple[str, int, str]] = []
        input_args: list[tuple[str, int]] = []
        output_args: list[tuple[str, int]] = []
        array_params: dict[int, int] = {}
        if method.portList is not None:
            for port in method.portList.ports:
                if "FunctionPort" not in str(port.kind):
                    continue
                pname = str(port.declarator.name.valueText)
                pwidth = _resolve_width(port.dataType)

                # Detect unpacked array dimensions (e.g. byte w_data[])
                decl = port.declarator
                if hasattr(decl, "dimensions") and decl.dimensions is not None:
                    for dim in decl.dimensions:
                        if "VariableDimension" in str(dim.kind):
                            array_params[len(args)] = pwidth
                            break

                # Determine direction: default is "input" per SV spec
                direction = "input"
                if hasattr(port, "direction") and port.direction is not None:
                    dir_str = repr(port.direction)
                    if "Output" in dir_str or "InOut" in dir_str:
                        direction = "output"

                args.append((pname, pwidth, direction))
                if direction == "output":
                    output_args.append((pname, pwidth))
                else:
                    input_args.append((pname, pwidth))

        dpi_funcs[func_name] = DpiFuncInfo(
            args=args,
            input_args=input_args,
            output_args=output_args,
            return_width=ret_width,
            unpacked_args_width=array_params,
        )

    return dpi_funcs, import_ranges


# ---------------------------------------------------------------------------
# Find all DPI call sites via recursive AST walk
# ---------------------------------------------------------------------------


def _find_invocations(
    node: pyslang.SyntaxNode,
    source: str,
    dpi_names: set[str],
    results: list[DpiCall],
) -> None:
    """
    Recursively walk the syntax tree and collect every InvocationExpression
    whose callee is a known DPI function.

    Each result: {func_name, arg_texts, offset_start, offset_end}
    """
    kind = str(node.kind)

    if kind == "SyntaxKind.InvocationExpression":
        # Check if the callee is a DPI function
        left = node.left
        fname = None
        if hasattr(left, "identifier"):
            fname = str(left.identifier.valueText)
        elif hasattr(left, "valueText"):
            fname = str(left.valueText)

        if fname and fname in dpi_names:
            # Extract argument source texts
            arg_texts = []
            for param in node.arguments.parameters:
                if "Argument" not in str(param.kind):
                    continue
                sr = param.sourceRange
                arg_texts.append(source[sr.start.offset : sr.end.offset].strip())

            sr = node.sourceRange
            results.append(
                DpiCall(
                    func_name=fname,
                    arg_texts=arg_texts,
                    start=sr.start.offset,
                    end=sr.end.offset,
                )
            )
            # Don't recurse into children of this invocation — we've captured it
            return

    # Generic child traversal: iterate all attributes that look like child nodes
    for attr in dir(node):
        if attr.startswith("_") or attr in _SKIP_ATTRS:
            continue
        try:
            val = getattr(node, attr)
        except Exception:
            continue
        if val is None or callable(val):
            continue
        _visit_val(val, source, dpi_names, results)


def _visit_val(
    val: pyslang.SyntaxNode,
    source: str,
    dpi_names: set[str],
    results: list[DpiCall],
) -> None:
    """Visit a value that might be a syntax node, a list of nodes, or a SyntaxList."""
    if hasattr(val, "kind"):
        # SyntaxList nodes are iterable containers — recurse into each child
        if "List" in str(val.kind):
            for item in val:
                if hasattr(item, "kind"):
                    _find_invocations(item, source, dpi_names, results)
        else:
            _find_invocations(val, source, dpi_names, results)
    elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            for item in val:
                if hasattr(item, "kind"):
                    _find_invocations(item, source, dpi_names, results)
        except TypeError:
            pass


_SKIP_ATTRS = frozenset(
    {
        "kind",
        "parent",
        "sourceRange",
        "to_json",
        "getFirstToken",
        "getLastToken",
        "isEquivalentTo",
        "visit",
        "text",
        "valueText",
        "rawText",
        "numericFlags",
        "withTrivia",
    }
)


def find_dpi_calls(
    tree: pyslang.SyntaxTree,
    source: str,
    dpi_names: set[str],
) -> list[DpiCall]:
    """Return all DPI call sites sorted by source offset (ascending)."""
    results: list[DpiCall] = []
    _find_invocations(tree.root, source, dpi_names, results)
    results.sort(key=lambda r: r.start)
    return results


# ---------------------------------------------------------------------------
# Source-level rewriting
# ---------------------------------------------------------------------------


def _find_module_body_range(source: str) -> tuple[int | None, int | None]:
    """Return (start, end) offsets of the first module's body (between header and endmodule)."""
    # Find the first ');' after 'module ... (' which closes the port list
    mod_match = re.search(r"\bmodule\b[^;]*\)\s*;", source)
    if not mod_match:
        return None, None
    body_start = mod_match.end()
    end_match = re.search(r"\bendmodule\b", source[body_start:])
    if not end_match:
        return body_start, len(source)
    return body_start, body_start + end_match.start()


def rewrite(
    source: str,
    dpi_funcs: dict[str, DpiFuncInfo],
    calls: list[DpiCall],
    import_ranges: list[tuple[int, int]],
) -> tuple[str, dict[str, int]]:
    """
    Rewrite source:
    1. For each DPI call (found on the *original* source), replace the call
       expression text with a wire name (or instantiation for void calls).
    2. Remove `import "DPI-C" ...` statements using AST source ranges.
    3. Insert wire declarations and blackbox instantiations at module level.

    Works on source offsets from the original parse, processing bottom-up so
    earlier offsets remain valid.

    Returns (rewritten_source, dpi_output_vars) where dpi_output_vars maps
    variable names driven by DPI cell output ports to their bit width.
    """
    result: str = source
    wire_decls: list[str] = []
    call_counts: dict[str, int] = {}
    dpi_output_vars: dict[str, int] = {}

    # Process calls bottom-up (reverse offset order) so replacements don't
    # invalidate earlier offsets.
    for call in reversed(calls):
        fname = call.func_name
        info = dpi_funcs[fname]

        count = call_counts.get(fname, 0)
        call_counts[fname] = count + 1
        suffix = (
            f"_{count}" if len([c for c in calls if c.func_name == fname]) > 1 else ""
        )
        wire_name = f"__dpi_{fname}_out{suffix}"
        inst_name = f"__dpi_{fname}{suffix}"

        has_ret = info.return_width is not None

        # Split call arg_texts by direction — input args become IN_n,
        # output args become OUT_n connections (directly to the lvalue).
        conns = []
        in_idx = 0
        out_idx = 0
        if has_ret:
            out_idx = 1  # OUT_0 reserved for return value wire
        for arg_text, (_pname, _pwidth, direction) in zip(call.arg_texts, info.args):
            if direction == "output":
                conns.append(f".OUT_{out_idx}({arg_text})")
                out_idx += 1
                dpi_output_vars[arg_text.strip()] = _pwidth
            else:
                conns.append(f".IN_{in_idx}({arg_text})")
                in_idx += 1
        if has_ret:
            conns.append(f".OUT_0({wire_name})")
        conn_str = ", ".join(conns)

        cell_name = _dpi_cell_name(fname, info)

        if has_ret:
            # Replace the call expression with the wire name
            result = result[: call.start] + wire_name + result[call.end :]

            # Queue wire declaration + instantiation for insertion
            assert info.return_width is not None
            w = info.return_width
            wire_decl = (
                f"    logic [{w - 1}:0] {wire_name};"
                if w > 1
                else f"    logic {wire_name};"
            )
            inst_line = f"    {cell_name} {inst_name} ({conn_str});"
            wire_decls.append(f"{wire_decl}\n{inst_line}")
        else:
            # Void call — remove the enclosing always block / expression statement
            # and place the instantiation at module level.
            end = call.end
            # Skip to end of statement (past semicolon)
            while end < len(result) and result[end] in " \t":
                end += 1
            if end < len(result) and result[end] == ";":
                end += 1

            # Try to find the enclosing always block and remove it entirely
            start = call.start
            # Search backwards for 'always' keyword on this or a previous line
            line_start = result.rfind("\n", 0, start) + 1
            search_start = max(0, line_start - 200)
            region = result[search_start:start]
            always_pos = region.rfind("always")
            if always_pos >= 0:
                # Remove from the 'always' keyword (including leading whitespace on that line)
                abs_always = search_start + always_pos
                always_line_start = result.rfind("\n", 0, abs_always) + 1
                # Also consume a trailing 'end' keyword left behind by a begin...end block
                tail_end = end
                tail_region = result[end : end + 30]
                m_end = re.match(r"[\s]*\bend\b[^\w]?", tail_region)
                if m_end:
                    tail_end = end + m_end.end()
                    # consume trailing newline
                    if tail_end < len(result) and result[tail_end] == "\n":
                        tail_end += 1
                result = result[:always_line_start] + result[tail_end:]
            else:
                # No enclosing always — just remove the statement
                result = result[:line_start] + result[end:]

            inst_line = f"    {cell_name} {inst_name} ({conn_str});"
            wire_decls.append(inst_line)

    # Remove DPI import statements using AST source ranges (bottom-up to
    # preserve earlier offsets).
    for start, end in sorted(import_ranges, reverse=True):
        result = result[:start] + result[end:]
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Insert wire declarations at the top of the module body
    if wire_decls:
        body_start, _ = _find_module_body_range(result)
        if body_start is not None:
            wire_decls.reverse()  # restore original order
            insert_block = "\n".join(wire_decls) + "\n"
            result = result[:body_start] + "\n" + insert_block + result[body_start:]

    return result, dpi_output_vars


# ---------------------------------------------------------------------------
# Blackbox generation
# ---------------------------------------------------------------------------


def _dpi_cell_name(func_name: str, info: DpiFuncInfo) -> str:
    """Build the width-encoded DPI cell name.

    Format: ``$__DPI__<func>$I<w0>_<w1>_...$O<w0>_<w1>_...``

    Input widths come from input_args only.  Output widths start with the
    return value (if non-void), followed by output-arg widths.
    """
    in_widths = "_".join(str(w) for _, w in info.input_args)
    out_parts = []
    if info.return_width is not None:
        out_parts.append(str(info.return_width))
    for _, w in info.output_args:
        out_parts.append(str(w))
    out_widths = "_".join(out_parts)
    return f"$__DPI__{func_name}$I{in_widths}$O{out_widths}"


def generate_blackbox(func_name: str, info: DpiFuncInfo) -> str:
    """Generate the (* blackbox *) module declaration for a DPI function.

    IN ports come from input_args only.
    OUT ports: OUT_0 from ret_width (if non-void), then OUT_N from output_args.
    """
    cell_name = _dpi_cell_name(func_name, info)
    has_out = info.return_width is not None or len(info.output_args) > 0
    # Void-only DPI (no outputs) needs (* keep *) so Yosys won't optimise
    # the sink cell away — the host callback still has side effects.
    attr = "(* blackbox *)" if has_out else "(* blackbox, keep *)"
    lines = [attr]
    ports = []
    for i, (_name, width) in enumerate(info.input_args):
        ports.append(
            f"    input  logic [{width - 1}:0] IN_{i}"
            if width > 1
            else f"    input  logic        IN_{i}"
        )
    out_idx = 0
    if info.return_width is not None:
        w = info.return_width
        ports.append(
            f"    output logic [{w - 1}:0] OUT_{out_idx}"
            if w > 1
            else f"    output logic        OUT_{out_idx}"
        )
        out_idx += 1
    for _name, width in info.output_args:
        ports.append(
            f"    output logic [{width - 1}:0] OUT_{out_idx}"
            if width > 1
            else f"    output logic        OUT_{out_idx}"
        )
        out_idx += 1
    lines.append(f"module {cell_name} (")
    lines.append(",\n".join(ports))
    lines.append(");")
    lines.append("endmodule")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fixup_synth_types(source: str) -> str:
    """Replace non-synthesizable SV types left in the module body.

    After DPI import removal and call rewriting, some non-synthesizable type
    keywords may remain in variable declarations.  This pass replaces them
    with their logic-vector equivalents so the output is accepted by Yosys'
    ``read_verilog -sv``.

    Handled patterns:
      chandle name  →  logic [63:0] name
      string  name  →  logic [63:0] name
      initial begin ... end  →  removed (contains $value$plusargs etc.)
    """
    # chandle → logic [63:0] (variable declarations only)
    source = re.sub(
        r"\bchandle\s+(\w+)",
        r"logic [63:0] \1",
        source,
    )

    # string variable declarations (not inside string literals or comments).
    # Match lines like:  string __uartlog;  or  string foo = ...;
    source = re.sub(
        r"^(\s*)string\s+(\w+)",
        r"\1logic [63:0] \2",
        source,
        flags=re.MULTILINE,
    )

    # Remove initial begin...end blocks (contain $value$plusargs,
    # DPI init calls that are now handled as DPI cells, etc.)
    source = re.sub(
        r"\binitial\s+begin\b.*?\bend\b\s*\n?",
        "",
        source,
        flags=re.DOTALL,
    )

    return source


def _fix_dpi_output_declarations(source: str, dpi_output_vars: dict[str, int]) -> str:
    """Fix declarations and assignments for variables driven by DPI cell outputs.

    When a void DPI call is converted to a module-level ``$__DPI__`` cell,
    the output parameter variables become cell-driven nets.  Any blocking
    assignments (typically in reset paths) create driver-driver conflicts in
    synthesis — Yosys picks the constant, permanently disabling the DPI output.

    This pass:
    1. Removes blocking assignments (``var = expr;``) to each DPI output var.
    2. Converts their declarations from ``bit``/``reg``/``int``/``byte``/``logic``
       to ``wire [width-1:0]`` so they are proper net types for structural
       connectivity.

    Non-blocking assignments (``<=``) to *other* variables that merely *read*
    the DPI output are left untouched.
    """
    if not dpi_output_vars:
        return source

    for var, width in dpi_output_vars.items():
        # 1. Remove blocking assignments:  var = expr;
        source = re.sub(
            rf"^\s*\b{re.escape(var)}\b\s*=[^;]+;\s*\n",
            "",
            source,
            flags=re.MULTILINE,
        )

        # 2. Convert declaration to wire.
        range_str = f" [{width - 1}:0]" if width > 1 else ""
        # Match: type_keyword [optional_range] var_name ;
        # Handles: bit __ready;  int __data;  logic [31:0] __data;  reg __foo;
        decl_pat = (
            rf"^(\s*)"
            rf"(?:{_TYPE_KW_PAT})"
            rf"(?:\s*\[\d+:\d+\])?"
            rf"\s+{re.escape(var)}\s*;"
        )
        source = re.sub(
            decl_pat,
            rf"\1wire{range_str} {var};",
            source,
            flags=re.MULTILINE,
        )

    return source


def preprocess(source: str) -> str:
    """Parse DPI imports, generate blackboxes, and rewrite the module."""
    tree = pyslang.SyntaxTree.fromText(source)
    dpi_funcs, import_ranges = parse_dpi_imports(tree, source)
    if not dpi_funcs:
        print("No DPI-C imports found.", file=sys.stderr)
        return source

    calls = find_dpi_calls(tree, source, set(dpi_funcs.keys()))

    # Resolve unpacked array parameter widths before generating blackboxes.
    converted = _resolve_array_params(source, dpi_funcs, calls)

    blackboxes = [generate_blackbox(name, info) for name, info in dpi_funcs.items()]
    rewritten, dpi_output_vars = rewrite(source, dpi_funcs, calls, import_ranges)
    rewritten = _fixup_synth_types(rewritten)

    # Replace unpacked array declarations with packed logic vectors.
    rewritten = _fixup_unpacked_arrays(rewritten, converted)

    # Fix declarations and remove blocking assignments for DPI output
    # variables that would conflict with the cell outputs.
    rewritten = _fix_dpi_output_declarations(rewritten, dpi_output_vars)

    header = "// Auto-generated by dpi_preprocess.py — do not edit."
    return "\n\n".join([header] + blackboxes + [rewritten.strip(), ""])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SV DPI-C imports into synthesizable blackboxes."
    )
    parser.add_argument("input", help="Input SystemVerilog file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        source = f.read()

    result = preprocess(source)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
