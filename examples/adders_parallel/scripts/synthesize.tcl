# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
if { $argc != 4 } {
    puts "Usage: yosys -c synthesize.tcl <input_sv> <output_gv> <lib_path> <N>"
    exit 1
}

set input_sv [lindex $argv 0]
set output_gv [lindex $argv 1]
set lib_path [lindex $argv 2]
set n [lindex $argv 3]

yosys read_verilog -sv $input_sv
yosys chparam -set N $n adders_parallel
yosys hierarchy -check -top adders_parallel

yosys proc
yosys opt_expr
yosys opt_dff
yosys opt_clean

yosys synth -flatten
yosys delete t:\$print

yosys dfflibmap -liberty $lib_path/aigpdk_nomem.lib
yosys opt_clean -purge
yosys abc -liberty $lib_path/aigpdk_nomem.lib
yosys opt_clean -purge
yosys abc -liberty $lib_path/aigpdk_nomem.lib
yosys opt_clean -purge
yosys write_verilog $output_gv
