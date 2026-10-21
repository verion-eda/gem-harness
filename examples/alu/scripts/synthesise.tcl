# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
if { $argc != 3 } {
    puts "Usage: yosys -p \"tcl synthesise.tcl <input_sv> <output_gv> <top_module>\""
    exit 1
}

set input_sv [lindex $argv 0]
set output_gv [lindex $argv 1]
set top_module [lindex $argv 2]

yosys read_verilog $input_sv
yosys hierarchy -check -top $top_module
yosys proc
yosys opt_expr
yosys opt_dff
yosys opt_clean

yosys synth -flatten
yosys delete t:\$print

yosys dfflibmap -liberty aigpdk/aigpdk_nomem.lib
yosys opt_clean -purge
yosys abc -liberty aigpdk/aigpdk_nomem.lib
yosys opt_clean -purge
yosys abc -liberty aigpdk/aigpdk_nomem.lib
yosys opt_clean -purge
yosys write_verilog $output_gv
