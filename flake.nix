# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# Nix flake for gem-harness development environment
# Provides Rust toolchain, Yosys (with slang plugin), and development tools
{
  description = "gem-harness";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs";
    # Isolated nixpkgs used only to pull a newer yosys (>=0.62). Kept separate
    # from the main nixpkgs so bumping yosys doesn't also bump gcc/other packages
    # that have broken the hypergraph partitioner in the past.
    nixpkgs-yosys.url = "github:NixOS/nixpkgs";
    rust-overlay.url = "github:oxalica/rust-overlay";
    yosys-slang = {
      url = "git+https://github.com/povik/yosys-slang?submodules=1";
      flake = false;
    };
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
      nixpkgs-yosys,
      rust-overlay,
      yosys-slang,
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [
          (import rust-overlay)
          (self: super: {
            rustToolchain = pkgs.symlinkJoin {
              name = "rust-toolchain";
              paths = [
                (super.rust-bin.stable.latest.minimal.override {
                  extensions = [
                    "clippy"
                    "rust-analyzer"
                    "rust-docs"
                    "rust-src"
                  ];
                })
                (super.rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))
              ];
            };
          })
          (import ./nix/cudatoolkit-no-stubs.nix)
        ];

        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        pkgsYosys = import nixpkgs-yosys { inherit system; };

        yosysSlang = pkgs.callPackage ./nix/yosys-slang.nix {
          yosys = pkgsYosys.yosys;
          src = yosys-slang;
        };
        yosysWithSlang = pkgs.symlinkJoin {
          name = "yosys-with-slang";
          paths = [ pkgsYosys.yosys ];
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/yosys \
              --add-flags "-m ${yosysSlang}/lib/slang.so"
          '';
        };

        nix-uv = pkgs.writeScriptBin "uv" ''
          #!${pkgs.zsh}/bin/zsh
          set -eu -o pipefail
          UV_PYTHON=${pkgs.python312}/bin/python ${pkgs.uv}/bin/uv --no-python-downloads $@
        '';

        clangdConfig = pkgs.writeText ".clangd" ''
          CompileFlags:
            Add:
              - --cuda-path=${pkgs.cudaPackages.cudatoolkit}
              - -L${pkgs.cudaPackages.cudatoolkit}/lib
              - -I${pkgs.cudaPackages.cudatoolkit}/include
        '';

        python_bins = [
          nix-uv
          pkgs.python312
          pkgs.ruff
          pkgs.basedpyright
          (pkgs.maturin.override { python3 = pkgs.python312; })
        ];

        common = with pkgs; [
          rustToolchain # Rust compiler and cargo
          pkg-config # Build-time dependency finder
          cargo-deny # Check dependencies for security issues
          cargo-edit # Cargo subcommands: add, rm, upgrade
          cargo-watch # Auto-rebuild on file changes
          cargo-machete # Check for unused dependencies
          cargo-insta # Regression tests with snapshot
          rust-cbindgen # Generate C FFI bindings
          pkgsCross.riscv32-embedded.buildPackages.gcc # RISC-V cross-compiler
          pkgsCross.riscv32-embedded.buildPackages.binutils # RISC-V cross-compiler
          yosysWithSlang # Yosys + SystemVerilog (slang) plugin
          zlib # Needed by Verilator FST tracing
          google-cloud-sdk # gsutil to download large artifacts
          pinact # pin gha
          just # just makefiles
          cmake
          hwloc
        ];

        cuda-deps = with pkgs; [
          cudaPackages.cuda_nvcc
          cudaPackages.cudatoolkit
        ];

        cuda-profiling = with pkgs; [
          cudaPackages.nsight_systems
          cudaPackages.nsight_compute
        ];

        libs = map (pkgs.lib.getOutput "lib") (
          with pkgs;
          [
            stdenv.cc.cc
            zlib
            glib
            eigen
          ]
        );

        commonEnv = {
          PYO3_PYTHON = "${pkgs.python312}/bin/python3";
        };

        commonUbuntuEnv = {
          LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu/";
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath libs}";
          CUDA_LIBRARY_PATH = "${pkgs.cudatoolkitNoStubs}";
        };

        commonShellHook = ''
          export CC="${pkgs.stdenv.cc}/bin/cc"
          export CXX="${pkgs.stdenv.cc}/bin/c++"
          echo "Yosys: $(yosys --version)"
          echo "Rust stable: $(${pkgs.rustToolchain}/bin/rustc --version)"
        '';

        # TODO not sure this holds - maybe Ubuntu should just use apt cuda drivers?
        # Surgically inject only libcuda.so.1 from the system driver without
        # polluting the full library search path (which would break Nix binaries).
        libcudaHook = ''
          if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
            export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.1''${LD_PRELOAD:+:$LD_PRELOAD}"
          fi
        '';

      in
      {
        devShells.default = pkgs.mkShell (
          commonEnv
          // {
            packages = python_bins ++ common;
            shellHook = commonShellHook;
          }
        );

        devShells.nixos = pkgs.mkShell (
          commonEnv
          // {
            packages =
              python_bins
              ++ common
              ++ libs
              ++ cuda-deps
              ++ cuda-profiling
              ++ [
                pkgs.lldb_20
                pkgs.clang-tools
              ];
            # NIXOS specific NVIDIA & CUDA
            LIBRARY_PATH = "/run/opengl-driver/lib/";
            LD_LIBRARY_PATH = "/run/opengl-driver/lib/:${pkgs.lib.makeLibraryPath libs}";
            CUDA_LIBRARY_PATH = "${pkgs.cudatoolkitNoStubs}";
            NSIGHT_COMPUTE_SECTIONS = "${pkgs.cudaPackages.nsight_compute}/sections";
            shellHook = commonShellHook + ''
              [ -f .clangd ] || cp ${clangdConfig} .clangd
            '';
          }
        );

        devShells.cuda_linux = pkgs.mkShell (
          commonEnv
          // commonUbuntuEnv
          // {
            packages = python_bins ++ common ++ libs ++ cuda-deps ++ cuda-profiling;
            NSIGHT_COMPUTE_SECTIONS = "${pkgs.cudaPackages.nsight_compute}/sections";
            shellHook = commonShellHook + libcudaHook;
          }
        );
      }
    );
}
