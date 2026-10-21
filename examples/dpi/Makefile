# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
.PHONY: all preprocess synth comp sim sim_c sim_dup sim_gpu sim_c_gpu bench clean
.PHONY: output_param preprocess_op synth_op comp_op sim_op
.PHONY: alltypes preprocess_at synth_at comp_at sim_at sim_c_at
.PHONY: const_input preprocess_ci synth_ci comp_ci sim_ci
.PHONY: test_array

ROOT     := $(abspath $(CURDIR)/../..)
DPI_DIR  := $(ROOT)/examples/dpi
BUILD_DIR := $(DPI_DIR)/build
BUILD_OP  := $(DPI_DIR)/build_output_param
BUILD_AT  := $(DPI_DIR)/build_alltypes
BUILD_CI  := $(DPI_DIR)/build_const_input
PYTHON   ?= uv run

all: sim

# ===== DPI Counter (original 32-bit example) =====

# Preprocess: replace DPI-C imports with $__DPI__ blackboxes
$(BUILD_DIR)/dpi_counter_synth.sv: rtl/dpi_counter.sv scripts/dpi_preprocess.py
	@echo "Preprocessing DPI imports..."
	mkdir -p $(BUILD_DIR)
	$(PYTHON) scripts/dpi_preprocess.py $< -o $@

preprocess: $(BUILD_DIR)/dpi_counter_synth.sv

# Synthesize: Yosys turns preprocessed SV → gatelevel.gv
$(BUILD_DIR)/gatelevel.gv: $(BUILD_DIR)/dpi_counter_synth.sv scripts/synthesize.tcl
	@echo "Synthesizing DPI counter..."
	cd $(DPI_DIR) && \
	    INPUT_FILE="$(BUILD_DIR)/dpi_counter_synth.sv" \
	    OUTPUT_FILE="$(BUILD_DIR)/gatelevel.gv" \
	    LIBERTY="$(ROOT)/aigpdk/aigpdk_nomem.lib" \
	    TOP_MODULE="dpi_counter" \
	    yosys -c scripts/synthesize.tcl

synth: $(BUILD_DIR)/gatelevel.gv

# Compile: compile → result.gemparts
$(BUILD_DIR)/result.gemparts: $(BUILD_DIR)/gatelevel.gv
	@echo "Compiling DPI counter..."
	cd $(ROOT) && cargo run -r -p compile --bin compile -- \
	    $(BUILD_DIR)/gatelevel.gv $(BUILD_DIR)/result.gemparts

comp: $(BUILD_DIR)/result.gemparts

# Simulate: run the Rust testbench on the CPU
sim: $(BUILD_DIR)/result.gemparts
	@echo "Simulating DPI counter..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_counter_testbench

# Simulate: run the Rust testbench on the GPU
sim_gpu: $(BUILD_DIR)/result.gemparts
	@echo "Simulating DPI counter (GPU)..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_counter_gpu_testbench --features cuda

# ===== DPI-C shared library =====

# Detect platform for shared library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  DPI_LIB_EXT := dylib
  SHARED_FLAG := -dynamiclib
else
  DPI_LIB_EXT := so
  SHARED_FLAG := -shared
endif

DPI_LIB     := $(BUILD_DIR)/libdpi.$(DPI_LIB_EXT)
DPI_LIB_DUP := $(BUILD_DIR)/libdpi_dup.$(DPI_LIB_EXT)

$(DPI_LIB): csrc/sat_add.c
	@echo "Compiling DPI-C shared library..."
	mkdir -p $(BUILD_DIR)
	$(CC) $(SHARED_FLAG) -o $@ $< -fPIC

$(DPI_LIB_DUP): csrc/sat_add_dup.c
	@echo "Compiling duplicate DPI-C shared library..."
	mkdir -p $(BUILD_DIR)
	$(CC) $(SHARED_FLAG) -o $@ $< -fPIC

# Simulate using the C DPI shared library
sim_c: $(BUILD_DIR)/result.gemparts $(DPI_LIB)
	@echo "Simulating DPI counter (C shared library)..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_counter_testbench -- --dpi-lib

# Simulate using the C DPI shared library on the GPU
sim_c_gpu: $(BUILD_DIR)/result.gemparts $(DPI_LIB)
	@echo "Simulating DPI counter (C shared library, GPU)..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_counter_gpu_testbench --features cuda -- --dpi-lib

# Test duplicate DPI function registration is caught
sim_dup: $(BUILD_DIR)/result.gemparts $(DPI_LIB) $(DPI_LIB_DUP)
	@echo "Testing duplicate DPI registration detection..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_dup_testbench

# ===== DPI Output Parameter (void function with output args) =====

$(BUILD_OP)/dpi_output_param_synth.sv: rtl/dpi_output_param.sv scripts/dpi_preprocess.py
	@echo "Preprocessing output-param DPI imports..."
	mkdir -p $(BUILD_OP)
	$(PYTHON) scripts/dpi_preprocess.py $< -o $@

preprocess_op: $(BUILD_OP)/dpi_output_param_synth.sv

$(BUILD_OP)/gatelevel.gv: $(BUILD_OP)/dpi_output_param_synth.sv scripts/synthesize.tcl
	@echo "Synthesizing output-param DPI..."
	cd $(DPI_DIR) && \
	    INPUT_FILE="$(BUILD_OP)/dpi_output_param_synth.sv" \
	    OUTPUT_FILE="$(BUILD_OP)/gatelevel.gv" \
	    LIBERTY="$(ROOT)/aigpdk/aigpdk_nomem.lib" \
	    TOP_MODULE="dpi_output_param" \
	    yosys -c scripts/synthesize.tcl

synth_op: $(BUILD_OP)/gatelevel.gv

$(BUILD_OP)/result.gemparts: $(BUILD_OP)/gatelevel.gv
	@echo "Compiling output-param DPI..."
	cd $(ROOT) && cargo run -r -p compile --bin compile -- \
	    $(BUILD_OP)/gatelevel.gv $(BUILD_OP)/result.gemparts

comp_op: $(BUILD_OP)/result.gemparts

sim_op: $(BUILD_OP)/result.gemparts
	@echo "Simulating output-param DPI..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_output_param_testbench

output_param: sim_op

# ===== DPI All-Types (all scalar types + multi-arg functions) =====

$(BUILD_AT)/dpi_alltypes_synth.sv: rtl/dpi_alltypes.sv scripts/dpi_preprocess.py
	@echo "Preprocessing alltypes DPI imports..."
	mkdir -p $(BUILD_AT)
	$(PYTHON) scripts/dpi_preprocess.py $< -o $@

preprocess_at: $(BUILD_AT)/dpi_alltypes_synth.sv

$(BUILD_AT)/gatelevel.gv: $(BUILD_AT)/dpi_alltypes_synth.sv scripts/synthesize.tcl
	@echo "Synthesizing alltypes DPI..."
	cd $(DPI_DIR) && \
	    INPUT_FILE="$(BUILD_AT)/dpi_alltypes_synth.sv" \
	    OUTPUT_FILE="$(BUILD_AT)/gatelevel.gv" \
	    LIBERTY="$(ROOT)/aigpdk/aigpdk_nomem.lib" \
	    TOP_MODULE="dpi_alltypes" \
	    yosys -c scripts/synthesize.tcl

synth_at: $(BUILD_AT)/gatelevel.gv

$(BUILD_AT)/result.gemparts: $(BUILD_AT)/gatelevel.gv
	@echo "Compiling alltypes DPI..."
	cd $(ROOT) && cargo run -r -p compile --bin compile -- \
	    $(BUILD_AT)/gatelevel.gv $(BUILD_AT)/result.gemparts

comp_at: $(BUILD_AT)/result.gemparts

sim_at: $(BUILD_AT)/result.gemparts
	@echo "Simulating alltypes DPI..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_alltypes_testbench

DPI_LIB_AT := $(BUILD_AT)/libdpi_alltypes.$(DPI_LIB_EXT)

$(DPI_LIB_AT): csrc/alltypes.c
	@echo "Compiling alltypes DPI-C shared library..."
	mkdir -p $(BUILD_AT)
	$(CC) $(SHARED_FLAG) -o $@ $< -fPIC

sim_c_at: $(BUILD_AT)/result.gemparts $(DPI_LIB_AT)
	@echo "Simulating alltypes DPI (C shared library)..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_alltypes_testbench -- --dpi-lib

alltypes: sim_at

# ===== DPI Constant Input (constant parameters to DPI) =====

$(BUILD_CI)/dpi_const_input_synth.sv: rtl/dpi_const_input.sv scripts/dpi_preprocess.py
	@echo "Preprocessing const-input DPI imports..."
	mkdir -p $(BUILD_CI)
	$(PYTHON) scripts/dpi_preprocess.py $< -o $@

preprocess_ci: $(BUILD_CI)/dpi_const_input_synth.sv

$(BUILD_CI)/gatelevel.gv: $(BUILD_CI)/dpi_const_input_synth.sv scripts/synthesize.tcl
	@echo "Synthesizing const-input DPI..."
	cd $(DPI_DIR) && \
	    INPUT_FILE="$(BUILD_CI)/dpi_const_input_synth.sv" \
	    OUTPUT_FILE="$(BUILD_CI)/gatelevel.gv" \
	    LIBERTY="$(ROOT)/aigpdk/aigpdk_nomem.lib" \
	    TOP_MODULE="dpi_const_input" \
	    yosys -c scripts/synthesize.tcl

synth_ci: $(BUILD_CI)/gatelevel.gv

$(BUILD_CI)/result.gemparts: $(BUILD_CI)/gatelevel.gv
	@echo "Compiling const-input DPI..."
	cd $(ROOT) && cargo run -r -p compile --bin compile -- \
	    $(BUILD_CI)/gatelevel.gv $(BUILD_CI)/result.gemparts

comp_ci: $(BUILD_CI)/result.gemparts

sim_ci: $(BUILD_CI)/result.gemparts
	@echo "Simulating const-input DPI..."
	cd $(ROOT) && cargo run -r -p examples --bin dpi_const_input_testbench

const_input: sim_ci

# ===== DPI Array (unpacked byte[] parameter preprocessing tests) =====

test_array:
	@echo "Running DPI array preprocessing tests..."
	cd $(ROOT) && uv run --script examples/dpi/scripts/test_dpi_array.py

bench: $(BUILD_DIR)/result.gemparts $(BUILD_DIR)/gatelevel.gv
	cd $(ROOT) && cargo run -r -p examples --features cuda --bin dpi_pipe_bench

clean:
	rm -rf $(BUILD_DIR) $(BUILD_OP) $(BUILD_AT) $(BUILD_CI)
