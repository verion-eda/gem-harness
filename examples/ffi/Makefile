# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# Makefile for FFI example
# Can be run from root with: make -C examples/ffi

ROOT_DIR = ../..
TARGET_DIR = $(ROOT_DIR)/target/release
BIN_DIR = $(ROOT_DIR)/bin
EXAMPLE_BIN = $(BIN_DIR)/ffi_example

CPP_SOURCE = ffi_example.cpp
HEADER = harness.h

# Test inputs
TEST_GATELEVEL = ../adder/build/gatelevel.gv
TEST_GEMPARTS = ../adder/build/result.gemparts

CXX = g++
CXXFLAGS = -DCUDA_ENABLED
LDFLAGS = -L$(TARGET_DIR) -lsimulator

.PHONY: all build-rust generate-header build-example run test clean

all: test

build-rust:
	cd $(ROOT_DIR) && cargo rustc -p simulator --release --lib --features "cuda"

generate-header:
	cd $(ROOT_DIR) && cbindgen --crate simulator --output examples/ffi/$(HEADER)

build-example: build-rust generate-header
	mkdir -p $(BIN_DIR)
	$(CXX) $(CPP_SOURCE) $(CXXFLAGS) $(LDFLAGS) -o $(EXAMPLE_BIN)

run: build-example
	LD_LIBRARY_PATH="$(TARGET_DIR):$$LD_LIBRARY_PATH" $(EXAMPLE_BIN) $(TEST_GATELEVEL) $(TEST_GEMPARTS)

test: run

clean:
	rm -f $(EXAMPLE_BIN)
	cd $(ROOT_DIR) && cargo clean
