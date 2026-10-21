//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.

#[cfg(feature = "cuda")]
use std::{env, path::Path};

fn main() {
    println!("Building cuda source files for GEM...");
    println!("cargo:rerun-if-changed=csrc");

    #[cfg(feature = "cuda")]
    {
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        // Let ucc auto-detect a host C++ compiler (nvcc --ccbin ...)
        cl_cuda.ccbin(false);
        // nvcc flags
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda
            .debug(false)
            .opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        // compile the kernel
        cl_cuda.compile("gemcu");
        let out_dir = env::var("OUT_DIR").unwrap();
        let gemcu_path =
            Path::new(&out_dir).join("ucc_cuda").join("libgemcu.a");
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&out_dir).join("ucc_cuda").display()
        );
        println!("cargo:rustc-link-arg={}", gemcu_path.display());
        println!("cargo:rustc-link-lib=dylib=cudart");
        // Generate the Rust bindings, by scanning the .cu file
        // for exported extern "C" functions and emit the bindings
        // to the rs file
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }
}
