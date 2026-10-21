# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# Shared recipes for gem-harness examples.
# Import with: import '../common.just'

# Compile a gatelevel netlist into a .gemparts file.
compile root gatelevel gemparts *flags:
    cd {{root}} && cargo run -r --bin compile -- {{gatelevel}} {{gemparts}} {{flags}}

# Run a examples testbench binary with CUDA.
run-testbench root bin *args:
    cd {{root}} && cargo run -r -p examples --features cuda --bin {{bin}} -- {{args}}
