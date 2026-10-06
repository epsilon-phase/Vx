#!/usr/bin/env bash
#===- level_zero_spike.sh - launch a SPIR-V image on an Intel GPU --------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Load a SPIR-V image into a Level Zero module, pass it Vx's argument list, run
# it on the GPU, and check the numbers. No oneAPI and no SYCL: Level Zero is
# the loader and the driver, and a C++ compiler with its headers is enough.
#
#   ./scripts/tools/level_zero_spike.sh
#
# What this is for, and what it is not. Vx's own device image does not validate
# yet (see the slice 4 note in docs/implementation_plans/xpu_backend_plan.md),
# and that failure must not be confused with the runtime being unproven. So this
# runs a hand-written kernel whose image does validate, in the exact parameter
# shape a Vx rank-2 tensor has: two pointers, an offset, two sizes, two strides,
# seven separate arguments and no aggregate.
#
# What a passing run establishes, on real hardware:
#
#   - the toolchain's SPIR-V loads through `zeModuleCreate` on this driver;
#   - the seven-value argument list is what a kernel launched this way wants;
#   - that list still describes the right elements when the tensor is a VIEW: a
#     row (a non-zero offset) and a column (a non-unit stride) each reach exactly
#     the elements they should and no others;
#   - allocate, copy in, launch, synchronize, copy back, and the values agree
#     with the host's.
#
# Observed on this machine (Arc A770, driver 12.55.8):
#
#   device: Intel(R) Arc(TM) A770 Graphics  (512 compute units)
#   memory module DDR: 15.11 GiB total
#   allocating 128 bytes of device memory (ceiling 12.0 GiB)
#   case whole   offset 0 sizes [8, 4] strides [4, 1]: correct
#   case row     offset 8 sizes [1, 4] strides [4, 1]: correct
#   case column  offset 1 sizes [8, 1] strides [4, 1]: correct
#
# The memory module's 15.11 GiB is the same figure the Arc's machine file records
# as usable, which is the cross-check that this is the card and not something
# behind it.
#
# The two API spellings that moved between Level Zero header versions are worth
# knowing before editing the C++: `ze_module_desc_t` wants `pInputModule` and
# `inputSize` here, and `ze_device_properties_t` publishes `maxMemAllocSize`,
# not a total-memory size -- that is a per-allocation ceiling, not the card's
# 16 GiB.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vx-level-zero-spike"
mkdir -p "$OUT"

for tool in clang spirv-val; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done
[ -d /usr/include/level_zero ] || {
  echo "error: no /usr/include/level_zero (install level-zero-headers)" >&2
  exit 1
}
# Asking the linker is the only check that matters: a table of cached libraries
# is not present on every machine, and the driver may sit in a directory the
# cache does not list.
printf 'int main() { return 0; }\n' >"$OUT/link_probe.cpp"
if ! "${CXX:-g++}" -std=c++17 "$OUT/link_probe.cpp" -lze_loader \
    -o "$OUT/link_probe" 2>"$OUT/link_probe.err"; then
  echo "error: cannot link against -lze_loader (install level-zero-loader)" >&2
  sed 's/^/  /' "$OUT/link_probe.err" >&2
  exit 1
fi

echo "==> building the image"
clang -target spirv64-unknown-unknown -x cl -cl-std=CL2.0 \
  -c "$HERE/level_zero_spike.cl" -o "$OUT/add_one.spv"
spirv-val "$OUT/add_one.spv"
echo "    $OUT/add_one.spv ($(wc -c <"$OUT/add_one.spv") bytes) validates"

echo "==> building the launcher"
"${CXX:-g++}" -std=c++17 -O2 -o "$OUT/spike" "$HERE/level_zero_spike.cpp" -lze_loader

echo "==> launching"
cd "$OUT"
./spike add_one.spv add_one
