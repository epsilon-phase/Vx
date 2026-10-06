#!/usr/bin/env bash
#===- spirv_module_check.sh - a Vx kernel through MLIR to SPIR-V ---------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Turn one `spawn on` region into a SPIR-V module with the in-tree MLIR, then
# ask the validator whether the module is any good. Needs no GPU and no vendor
# SDK: the whole route is MLIR passes and a serializer.
#
#   ./scripts/tools/spirv_module_check.sh [tests/backend/pass/<kernel>.vx]
#
# It runs the kernel twice:
#
#   plain  the module as it falls out of the route
#   priv   the same, with the kernel's own i32 stack slots marked private
#          before the gpu -> llvm-spv pass and unmarked after it
#
# The difference matters, because that marking is the one change known to remove
# a real error. A Vx kernel keeps its parallel-loop counters in `memref<i32>`
# slots, each of which becomes an `OpVariable` in the `Function` storage class
# whose type points into `CrossWorkgroup` -- which SPIR-V refuses:
#
#   error: Storage class must match result type storage class
#
# Marking the slots private clears that. It does not make the module valid: the
# next error is a memref descriptor whose pointer fields come out typed `i8`
# while the pointers that fill them are `float*` or `i32*`, and the validator
# rejects the composite that builds it:
#
#   error: The Object type (OpTypePointer) does not match the type that results
#          from indexing into the Composite (OpTypePointer)
#
# So today both runs print a rejection, and this script is how that stays
# checkable rather than a paragraph in a document. When the lowering stops
# putting memref values on the device side -- loop counters in SSA values
# instead of slots, or a row view instead of `memref.reinterpret_cast` -- this
# script is what says the module validates. That is why the verdict is the
# output: a size or an exit code would not tell the two apart.
#
# Three things had to be right, and each was a wrong turn first:
#
#   1. Only the `gpu.module` block goes into the pipeline. The host half of the
#      same file still has `vx.` ops, and `mlir-opt` cannot even parse it.
#   2. The address space marking has to go on before the gpu -> llvm-spv pass
#      and come off after it. Marking afterwards is too late: the descriptors
#      built around the slots already carry the old storage class.
#   3. Address space 5 is not "private" on this target -- it is CrossWorkgroup,
#      the storage class in the error above. Marking the slots with 5 is how an
#      earlier attempt at this concluded, wrongly, that the marking does not
#      work.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

VX_FILE="${1:-tests/backend/pass/placed_kernel_four_operands.vx}"
VXC="${VXC:-target/release/vxc}"
OUT="${TMPDIR:-/tmp}/vx-spirv-check"
mkdir -p "$OUT"

for tool in mlir-opt mlir-translate llc spirv-val; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool not on PATH (source config.local)" >&2
    exit 1
  }
done
[ -x "$VXC" ] || VXC="./$VXC"
[ -x "$VXC" ] || {
  echo "error: no vxc at $VXC (cargo build --release)" >&2
  exit 1
}

echo "==> $VX_FILE"
"$VXC" "$VX_FILE" --emit-mlir \
  -X mlir=--pass-pipeline="builtin.module(convert-vx-to-standard)" \
  >"$OUT/std.mlir" 2>/dev/null

# Keep only the kernel module: that block is the device image, and it is the
# only part a plain mlir-opt can read.
python3 - "$OUT/std.mlir" "$OUT/kernel.mlir" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
start = next((i for i, l in enumerate(lines)
              if re.match(r"^\s*gpu\.module\s+@vx_kernels\b", l)), None)
if start is None:
    sys.exit("no gpu.module @vx_kernels in the module -- is there a kernel here?")
body, depth, opened = [], 0, False
for line in lines[start:]:
    depth += line.count("{") - line.count("}")
    body.append(line)
    if opened and depth <= 0:
        break
    opened = True
if len(body) < 2:
    sys.exit("the gpu.module is empty: the kernel was dropped before this point")
indent = min(len(l) - len(l.lstrip()) for l in body[1:] if l.strip())
out = [body[0].strip()] + [l[indent:] if l.strip() else "" for l in body[1:]]
open(dst, "w").write("module {\n" + "\n".join(out) + "\n}\n")
PY

echo "    kernel module: $(wc -c <"$OUT/kernel.mlir") bytes"

# The slots to mark: the kernel's own i32 stack memory, and nothing outside it.
# A slot is one `memref.alloca`; the type name itself appears once per use.
echo "    i32 stack slots in the kernel: $(grep -c 'memref\.alloca() : memref<i32>' "$OUT/kernel.mlir" || true)"

for variant in plain priv; do
  in="$OUT/kernel.mlir"
  if [ "$variant" = priv ]; then
    in="$OUT/kernel_priv.mlir"
    sed 's/memref<i32>/memref<i32, #gpu.address_space<private>>/g' \
      "$OUT/kernel.mlir" >"$in"
  fi

  echo "==> $variant: gpu -> llvm-spv, hoist, llvm, serialize"
  mlir-opt --pass-pipeline='builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-llvm-spv{use-64bit-index=true}))' \
    "$in" >"$OUT/$variant.stage2.mlir"

  if [ "$variant" = priv ]; then
    # Off again, or the memref -> llvm conversion has an attribute it does not
    # know what to do with.
    sed 's/, #gpu.address_space<private>//g' "$OUT/$variant.stage2.mlir" \
      >"$OUT/$variant.stage3.mlir"
    mv "$OUT/$variant.stage3.mlir" "$OUT/$variant.stage2.mlir"
  fi

  # The gpu.module wrapper is not translatable, so its body moves up to module
  # scope, where the target triple goes on it.
  python3 - "$OUT/$variant.stage2.mlir" "$OUT/$variant.hoisted.mlir" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
start = next((i for i, l in enumerate(lines)
              if re.match(r"^\s*gpu\.module\s+@vx_kernels\b", l)), None)
if start is None:
    sys.exit("the gpu.module vanished between passes")
body, depth, opened = [], 0, False
for line in lines[start:]:
    depth += line.count("{") - line.count("}")
    if not opened:
        opened = True
        continue
    if depth <= 0:
        break
    body.append(line)
if not body:
    sys.exit("the gpu.module is empty after the gpu -> llvm-spv pass")
indent = min(len(l) - len(l.lstrip()) for l in body if l.strip())
out = [l[indent:] if l.strip() else "" for l in body]
open(dst, "w").write(
    'module attributes {llvm.target_triple = "spirv64-unknown-unknown"} {\n'
    + "\n".join(out) + "\n}\n")
PY

  mlir-opt --pass-pipeline='builtin.module(convert-to-llvm,reconcile-unrealized-casts)' \
    "$OUT/$variant.hoisted.mlir" >"$OUT/$variant.llvm.mlir"
  mlir-translate --mlir-to-llvmir "$OUT/$variant.llvm.mlir" >"$OUT/$variant.ll"
  llc -mtriple=spirv64-unknown-unknown -filetype=obj "$OUT/$variant.ll" \
    -o "$OUT/$variant.spv"

  echo "    image: $(wc -c <"$OUT/$variant.spv") bytes"
  if command -v spirv-dis >/dev/null 2>&1; then
    n=$(spirv-dis "$OUT/$variant.spv" | grep -c 'OpFunctionParameter' || true)
    echo "    kernel parameters: $n"
  fi

  echo -n "    spirv-val: "
  if spirv-val "$OUT/$variant.spv" 2>"$OUT/$variant.val.err"; then
    echo "VALID"
  else
    echo "rejected --"
    sed 's/^/      /' "$OUT/$variant.val.err"
  fi
done

cat <<EOF

Artifacts: $OUT
Both variants are rejected today, and the two errors are different ones: the
storage-class error on the slots clears when they are marked private, and what
is left is the pointer type inside a memref descriptor. See the header.
EOF
