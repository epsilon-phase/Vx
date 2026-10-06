#!/usr/bin/env bash
#===- onemkl_gemm_spike.sh - a real GEMM on the Intel GPU -----------------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Build and run the oneMKL GEMM spike: a square fp32 GEMM on the GPU, verified
# against a host reference at 256, then timed at the size given (8192 by
# default). Prints TFLOPS.
#
#   ./scripts/tools/onemkl_gemm_spike.sh [size]
#
# This one needs the oneAPI toolchain, unlike the Level Zero spike beside it:
# oneMKL's SYCL interface is built with DPC++ and its headers are C++. That is
# the same reason `docs/gpu_backends.md` gives for the dispatch library being
# built with oneAPI's compiler -- the vendor library needs a queue, and the queue
# needs SYCL.
#
# Memory: three fp32 matrices of the timed size, which is 786 MiB at 8192, and
# the program refuses to go past `VX_VRAM_CEILING` (12 GiB by default) because
# this machine is unstable above 12 GiB of VRAM in use.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIZE="${1:-8192}"
OUT="${TMPDIR:-/tmp}/vx-onemkl-gemm"
mkdir -p "$OUT"

ONEAPI=/opt/intel/oneapi
[ -f "$ONEAPI/setvars.sh" ] || {
  echo "error: no $ONEAPI/setvars.sh (install the oneAPI toolkit)" >&2
  exit 1
}
# The environment script is written to be sourced from a shell that tolerates
# unset variables, and it RETURNS NON-ZERO on success (3 here), which `set -e`
# turns into a silent exit before anything is built. So the return code is
# ignored and the check below is what judges whether it worked.
set +u
set +e
# shellcheck disable=SC1091
source "$ONEAPI/setvars.sh" >"$OUT/setvars.log" 2>&1
set -e
set -u

command -v icpx >/dev/null 2>&1 || {
  echo "error: icpx not on PATH after sourcing setvars.sh" >&2
  exit 1
}
[ -n "${MKLROOT:-}" ] || {
  echo "error: MKLROOT unset after sourcing setvars.sh" >&2
  exit 1
}

echo "==> building"
icpx -fsycl -O2 -I"$MKLROOT/include" \
  "$HERE/onemkl_gemm_spike.cpp" -o "$OUT/gemm" \
  -L"$MKLROOT/lib" -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
  -lsycl -lOpenCL -lpthread -lm -ldl

echo "==> running"
cd "$OUT"
./gemm "$SIZE"
