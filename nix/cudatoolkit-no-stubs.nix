# SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
# SPDX-License-Identifier: Apache-2.0
# CUDA toolkit without stub libraries
#
# Problem: When building Rust crates that link to CUDA, the cc crate automatically
# adds all discovered CUDA library paths to RPATH in the compiled binary. The CUDA
# toolkit includes stub libraries (in lib/stubs/ or as direct symlinks pointing to
# /stubs/ paths) which are meant only for linking, not runtime. At runtime, RPATH
# is checked before LD_LIBRARY_PATH, so if stubs are in RPATH, the stub libcuda.so
# gets loaded instead of the real NVIDIA driver at /run/opengl-driver/lib/libcuda.so.
# The stub library causes CUDA initialization (cuInit) to fail with "UnknownError".
#
# Attempted solution: Create a filtered CUDA toolkit that excludes stub libraries, then
# point CUDA_ROOT/CUDA_PATH to this filtered version. Stubs remain available for
# linking via LIBRARY_PATH. However, the build system may still find and add stub paths
# to RPATH from other sources, requiring post-build RPATH patching as a fallback.

# NOTE this does not work:
# cudatoolkitNoStubs = super.symlinkJoin {
#   name = "cudatoolkit-no-stubs";
#   paths = [ super.cudaPackages.cudatoolkit ];
#   postBuild = ''
#     # find_cuda_helper expects lib64, but NixOS uses lib
#     ln -s $out/lib $out/lib64
#   '';
# };

# Overlay function
self: super:
{
  cudatoolkitNoStubs = super.runCommand "cudatoolkit-no-stubs" { } ''
    mkdir -p $out/lib $out/include

    # copy all non-stub libraries
    for item in ${super.cudaPackages.cudatoolkit}/lib/*; do
      if [ "$(basename "$item")" != "stubs" ]; then
        if [ -e "$item" ]; then
          resolved=$(readlink -f "$item" 2>/dev/null || echo "$item")
          # Skip if resolved path contains /stubs/
          if [[ "$resolved" != *"/stubs/"* ]]; then
            ln -s "$item" "$out/lib/$(basename "$item")"
          fi
        fi
      fi
    done

    ln -s ${super.cudaPackages.cudatoolkit}/include $out/include

    # workaround for rust cuda crates only checking lib64
    ln -s $out/lib $out/lib64
  '';
}
