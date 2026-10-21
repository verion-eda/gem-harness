# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# Minimal yosys-slang derivation. Adapted from lowRISC/ibex.
# Upstream src is passed in via the flake input with submodules=1.
{
  lib,
  stdenv,
  cmake,
  python3,
  yosys,
  src,
}:
stdenv.mkDerivation {
  inherit src;
  pname = "yosys-slang";
  version = "unstable";

  nativeBuildInputs = [
    cmake
    python3
  ];
  buildInputs = [ yosys ];

  cmakeFlags = [ "-DYOSYS_CONFIG=${yosys}/bin/yosys-config" ];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp slang.so $out/lib/slang.so
    runHook postInstall
  '';

  meta = {
    description = "SystemVerilog frontend for Yosys";
    homepage = "https://github.com/povik/yosys-slang";
    license = lib.licenses.mit;
    platforms = lib.platforms.all;
  };
}
