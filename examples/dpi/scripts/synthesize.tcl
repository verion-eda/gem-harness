# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# Yosys TCL script: run with yosys -c synthesize.tcl
# Pass paths via environment: INPUT_FILE, OUTPUT_FILE, optionally TOP_MODULE, LIBERTY
yosys -import

set script_dir [file dirname [info script]]
if {[info exists env(INPUT_FILE)]}  { set input_file  $env(INPUT_FILE)  } else { set input_file  "" }
if {[info exists env(OUTPUT_FILE)]} { set output_file $env(OUTPUT_FILE) } else { set output_file "" }
if {[info exists env(TOP_MODULE)]}  { set top         $env(TOP_MODULE)  } else { set top         "" }
if {[info exists env(LIBERTY)]}     { set liberty     $env(LIBERTY)     } else { set liberty [file join $script_dir "aigpdk" "aigpdk_nomem.lib"] }

if {$input_file eq ""} {
    puts stderr "Error: INPUT_FILE environment variable is required"
    exit 1
}
if {$output_file eq ""} {
    puts stderr "Error: OUTPUT_FILE environment variable is required"
    exit 1
}

read_verilog -sv $input_file
if {$top ne ""} {
    hierarchy -check -top $top
} else {
    hierarchy -check -auto-top
}
procs;;
opt_expr; opt_dff; opt_clean
synth -flatten
delete t:\$print
dfflibmap -liberty $liberty
opt_clean -purge
abc -liberty $liberty
opt_clean -purge
abc -liberty $liberty
opt_clean -purge
write_verilog $output_file
