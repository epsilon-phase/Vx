# Intel XPU (Arc A770) backend: implementation plan

**Status:** in progress on branch `xpu-backend`. Tracking issue: #1137. The
roadmap this plan follows is `docs/gpu_backends.md` (PR #1138), which is not
merged yet; read it from the PR branch. That document says *what* a backend is
and *why* the policies are what they are. This plan only records where this
tree stands today, what lands in what order, and what the roadmap leaves
undecided — it does not restate the roadmap.

______________________________________________________________________

## 1. Where this starts from

A backend is three pieces (the roadmap's own split):

1. **A machine file** — the hardware declared in `.vx`.
1. **A device image compiler** — `spawn on(...)` outlined into a standard
   `gpu.module`, then compiled to the device's image format by a sibling of
   `deviceImageOf()` in `src/dialect/VxLowering.cpp:2673`.
1. **A runtime dispatch library** — a C file beside `runtime/cuda_dispatch.cpp`
   implementing the `vx_plugin_*` entry points in
   `include/vx_hardware_runtime.h`, plus an arm in `build.rs`.

This plan lands them in that order, and the roadmap's rule decides what goes
first within it: **the CI-testable half lands before any runtime exists.** The
device-image half compiles and its output is checked with FileCheck on a
machine with no GPU and no vendor SDK, so it can merge on its own; the runtime
comes after. Each slice below is one PR; dependent PRs go in a GitHub stacked
PR, and CI checks the CLA on the first one (`docs/CLA.md`).

The six shared work items in §2 are not a fourth piece. They are places where
the compiler currently assumes NVIDIA is the only device target; each is
vendor-neutral and gets its own issue, so they keep their value whichever
backend lands first.

______________________________________________________________________

## 2. The six shared work items, located in this tree

Each item from the roadmap's "Shared work before a second backend", with the
code site it names, read from this branch. The last line of each entry says
whether this plan's first slices cover it. Line numbers were read from this
branch while the §3 slices were landing (commit 886a27ec plus the working
tree); once an in-flight slice has landed, match its sites by the symbol names
given here. These commits were rebased from ea86415c onto 3d59497a after main
was force-pushed, so a hash quoted in an older note or a chat session is a
pre-rebase hash -- the reflog keeps those until it is pruned.

**1. The kernel eligibility gate accepts one arch.**
`materializeGpuKernels` at `src/dialect/VxLowering.cpp:1419` decides which
kernels get cloned into the `gpu.module`. The single-arch test is
`src/dialect/VxLowering.cpp:1436-1438` — `if (arch.getValue() != "nvptx64")
return;` — with the dispatch-id band as the fallback at `:1439` for kernels
that carry no `arch` attribute.
*Landed as a7b44472*: `DevicePipeline` with `pipelineForArch` as the table, keyed
on the declared arch, `nvptx64` as its only row and the band fallback kept.

*What one module per arch drags in, raised in upstream's review of this work:*
the pass does not only clone the kernels into the `gpu.module`; it also copies
what they use. Verified at the pull-request head, not in this tree (this branch's
base predates #1151, so the code is not here): the copies are built at
`src/dialect/VxLowering.cpp:1519` onward, and a static shared-memory alloca
becomes a module-level `memref.global` at `:1593`. With one module per
architecture, a helper or a table that two architectures' kernels both use has to
be copied into each module — and deduplicated per module, once each. Today's
single-arch code hides the question: there is one module to deduplicate within.
The exact `copied`-set spelling in that revision is not checkable from here.

**2. The dispatch payload cannot carry a binary image.**
The payload is built at `src/dialect/VxLowering.cpp:2157-2265`: kernel name
first, then NUL-separated `key=value` entries (`kind=` at `:2166`, `topo=` at
`:2186`, `launch=` at `:2210`, ...), with `image=` appended last at `:2261`. The text-only
assumption is enforced inside `deviceImageOf` at `src/dialect/VxLowering.cpp:2755-2757`:
an image containing a NUL byte is rejected. The reader is `vx_payload_field` at
`include/vx_hardware_runtime.h:250`, which walks entries and answers only
the key it is asked for — that property is what lets new entries land without
breaking existing consumers (`runtime/cuda_dispatch.cpp:699` reads `image=` this
way).
*Landed as c57fe069.* The shape: `abi=1` as a version key, plus an
`imagebin=` entry carrying a base64-encoded image for formats that are not text.
No producer emits `imagebin=` until the device-image slice has a binary format
to put there.

**3. Address spaces are mapped for NVVM only.**
The NVPTX numbering (0 generic, 1 global, 3 shared, 5 local) lives in one
function, `AddressSpace::nvptx_addrspace` at `src/arch.rs:470-477` (its
original test still pins 0/1/3/5 at `:1547-1550`), with the symbolic
`gpu_attr` spelling beside it at `:482-489`. Before this branch's slice the
number was wired raw into codegen: `generator.rs`'s two call sites spelled it
out, and `tensor.rs` hardcoded four `, 3` shared-memory suffixes. On the C++ side,
`src/dialect/VxLowering.cpp` still hardcodes NVVM's 3 in three places — the
integer written into the device memref type at `:977`, and the two
`space.getInt() == 3` checks at `:1492` and `:1572`.
*Landed as abd6eedf, Rust half*: a `DeviceTarget` table at `src/arch.rs:500-543` keyed
on the declared arch — `from_arch` at `:518` recognizes
`x86_64`/`aarch64`/`nvptx64`/`amdgcn`/`spirv64` — answered through
`address_space_for_arch` (`:549-553`), routed through `generator.rs:1702` and
`:2066` and a new `shared_addrspace()` in
`src/codegen/flat/emit/tensor.rs:23` (call sites `:132`, `:267`, `:860`,
`:957`). Deliberately no SPIR-V arm: every target other than `nvptx64` answers
today's numbers until its pipeline brings the table — those numbers come from
MLIR's storage-class conversion, not from a guess. The three `VxLowering.cpp`
literals remain, and follow with the device-image slice.

**4. Scalar element types escape the `dtypes:` check.**
`check_element_type` at `src/hir/check/transfer.rs:389-425` produces E6026, and
it is reached for placed tensors at `:907` and transferred tensors at `:1082`.
A `spawn` body goes through `check_spawnon_expr` at
`src/hir/check/transfer.rs:1875`, which resolves the target topology but never
compares the body's element types against that topology's `dtypes:` list — so
an f64 scalar inside a `spawn` body passes the checker and would only fail on
the device.
*Landed as e9fbd3b3, at the binding rather than over the whole body. A value that
appears solely as the region's result is still open.*

**5. One dispatch library per build.**
`build.rs:449-454` picks exactly one dispatch source — the CUDA library where a
toolkit is installed, otherwise a host shim by architecture — and the program
loads that one, `VX_DISPATCH_LIB` overriding by hand (`src/jit.rs:164-167` and
`:364-365`).
*Not covered by an in-flight slice.* The runtime slice adds an arm in this same
shape and defers topology-routed dispatch (see §3 slice 6 and §6 question 6).

**6. A conformance test set.**
The reference shape exists: `tests/backend/pass/placed_kernel_four_operands.vx`
— a placed region no vendor library can route, four rank-2 operands — with
`tests/integration_test/device_image_test.rs` asserting today's PTX in the
payload and `tests/runtime/kernel_launch_test.cpp` asserting the matching
parameter list (28 parameters for that signature).
*Not covered by an in-flight slice.* The parity slice builds the vendor-neutral
set on this shape (§3 slice 7).

______________________________________________________________________

## 3. The slices

Each slice is small enough for one PR. Owner for every slice is the
contributor driving #1137 unless the slice says otherwise; the parity slice
additionally needs the hardware owner — the person with the card, whose
measurements are recorded in `fleet/arc-a770.vx` (2026-10-04). Every "proves"
line is a claim about a machine with no GPU and no vendor SDK — that is what
CI is — except slice 7, whose whole point is the card.

### Slice 0 — the machine file (done)

*Change.* `fleet/arc-a770.vx` declares the part: 16 GiB HBM, 16 MiB L2, 64 KiB
SMEM, `arch: spirv64` (`:74`), `dtypes:` without f64 (`:86`), both PCIe
directions. Provenance rows added to `fleet/README.md`. Landed on this branch as
commit 886a27ec; no compiler change, which is the roadmap's point about step 1.
A region spawned on this topology takes the host fallback today, or the honest
refusal when its operands sit in memory the host cannot read.

*Proves.* The existing check shapes, run against `--machine fleet/arc-a770.vx`:
the capacity check refuses a working set above 16 GiB (E6009, the shape of
`tests/frontend/fail/a_device_spelled_placement_is_admitted.vx`), and an f64
tensor placed there reports E6026 naming the declared list (the shape of
`tests/frontend/fail/dtype_not_on_this_machine.vx`). **Landed as d9822b2b:** the two
`MATRIX` rows in `tests/integration_test/fleet_dtype_test.rs` are now in CI
(`("arc-a770", "f64", false)`, `("arc-a770", "i4", true)`,
`("arc-a770", "f32", true)`, `("arc-a770", "f16", true)`), each checked by
flipping its expectation and reading which row the report named. That commit also
added the INT4 matrix format to `dtypes:`, which the file's provenance row in
`fleet/README.md` had cited all along.

*Deferred.* Everything about execution; `replicas:`/`clock:` on SMEM (which
unit a workgroup's scratchpad belongs to on Xe is unsettled and a wrong value
moves every on-die cost); the UNVERIFIED bandwidth figures.

### Slice 1 — eligibility gate keyed on arch (landed as a7b44472; shared item 1)

*Change.* `src/dialect/VxLowering.cpp:1436-1438` becomes a table keyed on the
declared arch: `nvptx64` as its only entry, band fallback kept for kernels with
no arch attribute. An arch with no pipeline answers "no" here, the way
`applegpu` already does.

*Proves.* The existing device-image fixture
(`tests/backend/pass/custom_topology_device_image.vx`) still passes, and the
emitted MLIR of the whole fixture corpus is unchanged — for every program that
names `nvptx64` the gate's answer is what it was.

*Deferred.* The `spirv64` entry, which arrives with slice 4 when there is a
pipeline behind it.

### Slice 2 — payload version key and binary image entry (landed as c57fe069; shared item 2)

*Change.* The payload builder at `src/dialect/VxLowering.cpp:2157` gains an
`abi=1` entry and, for an image that is not text, an `imagebin=` entry carrying
it base64-encoded (decode helpers beside `vx_payload_field` in the header);
the dispatch side refuses an `abi=` it does not know instead of guessing.
`vx_payload_field` itself needs no change: it already walks past entries
nobody asked for.

*Proves.* A new unit test, `tests/runtime/payload_test.cpp` with its Rust
driver `tests/integration_test/payload_test.rs`: decode round-trip, a payload
with no `abi=` still readable, malformed base64 refused. And reading
`vx_payload_field` plus the CUDA payload reads
(`runtime/cuda_dispatch.cpp:699, :766, :825, :938`) shows every existing
consumer asks for specific keys, so `abi=1` and `imagebin=` are invisible to
them. No GPU, no SDK.

*Deferred.* Freezing the `vx_plugin_*` interface — the roadmap says it is not
frozen, and the version key is precisely what makes changing it safe later.

### Slice 3 — address-space table, Rust half (landed as abd6eedf; shared item 3)

*Change.* Landed as abd6eedf: one `DeviceTarget`
address-space table in `src/arch.rs:500-543`, keyed on the declared arch from
`EmitCtx.topo_archs`. It replaces `generator.rs`'s two direct `nvptx_addrspace()`
calls (now `address_space_for_arch` at `src/codegen/generator.rs:1702` and
`:2066`) and `tensor.rs`'s four hardcoded `, 3>` suffixes (now routed through
`shared_addrspace()`, `src/codegen/flat/emit/tensor.rs:23`).

*Proves.* The unit test `address_space_numbers_come_from_the_declared_arch`
(`src/arch.rs:1570-1608`): the NVPTX arm answers 0/1/3/5 (`:1582`), an absent
or unrecognized arch answers today's numbers rather than failing or
guessing (`:1597-1598`), and `from_arch` recognizes
`x86_64`/`aarch64`/`nvptx64`/`amdgcn`/`spirv64` (`:1586-1591`); plus an
emitted-MLIR diff over every `.vx` fixture in the tree, which must be
byte-identical — the bar for a refactor with no intended behavior change, and
an instance of the repo's rule that emitted MLIR stays byte-identical run to
run.

*Deferred.* The SPIR-V arm of the table (it comes from MLIR's storage-class
conversion together with the pipeline); the three integer-3 literals in
`VxLowering.cpp` (`:977`, `:1492`, `:1572`), which stay NVVM's until the
SPIR-V pipeline installs its own conversion.

### Slice 4 — the SPIR-V device image (blocked: no route yields a valid module yet)

*Investigation first.* No repo files — a reproducible command sequence proving
the route `gpu.module` → SPIR-V with the in-tree MLIR (convert-gpu-to-spirv,
ABI-attribute lowering, serialization, spirv-val), plus a ranked list of
blockers, recorded on #1137. This is the roadmap's "How to start" step 3, and
its output decides §6 questions 1 and 3.

*Change.* A sibling of `deviceImageOf` (`src/dialect/VxLowering.cpp:2673`),
selected by the module's declared arch: `nvptx64` keeps today's NVVM/PTX
pipeline, `spirv64` runs the SPIR-V one. The gate table from slice 1 gains its
`spirv64` entry. The image reaches the payload through slice 2's `imagebin=`
entry — binary formats never touch the `image=`/NUL path.

*Proves.* A fixture under `tests/backend/` whose RUN line pipes
`vxc %s --action emit-mlir` into FileCheck, in the shape of
`tests/backend/pass/custom_topology_device_image.vx:8-26`, asserting the
`spirv64` spawn carries an image; plus an assertion beside
`tests/integration_test/device_image_test.rs` that the decoded bytes carry the
SPIR-V magic number, because an empty or truncated image would satisfy a
presence check and then fail at the runtime, a machine away from the cause.

*Deferred.* Running the image (slice 6); matmul; which route is used, reopened by
the 2026-10-06 measurement below.

**Investigation result — 2026-10-04, corrected 2026-10-06, MLIR 22.1.8.** Run end
to end on this machine, from a real kernel. The LLVM route produces a module; it
does not yet produce a module `spirv-val` accepts, for every kernel tried. That
is a different blocker from the one this note first named, and it is not in the
kernel's parameter list.

*The device twin as it exists.* `vx-opt --convert-vx-to-standard` turns a
`vx.spawn` into `vx.kernel` + `vx.launch`, and materializes
`gpu.module @vx_kernels { gpu.func @vx_npu_kernel_0(memref<8x4xf32>, ...) }` --
static shapes, no address-space annotations, and a **CFG** (`cf.br`,
`cf.cond_br`) that comes from the frontend, not from a device pass.

*The route, five steps, and a script now runs it.* `scripts/tools/spirv_module_check.sh`
takes a `.vx` file and prints the validator's verdict for each variant.

1. `vxc --emit-mlir -X mlir=--pass-pipeline="builtin.module(convert-vx-to-standard)"`,
   then keep only the `gpu.module @vx_kernels` block: the host half of the same
   file still has `vx.` ops, which a plain `mlir-opt` cannot parse, and it is not
   part of the device image.
2. `mlir-opt --pass-pipeline='builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-llvm-spv{use-64bit-index=true}))'`.
3. Hoist the `gpu.module` body to module scope and set
   `module attributes {llvm.target_triple = "spirv64-unknown-unknown"}`.
4. `mlir-opt --convert-to-llvm --reconcile-unrealized-casts` (the same reconcile
   the NVVM pipeline runs; without it the translation dies on a
   `builtin.unrealized_conversion_cast` from descriptor struct to memref), then
   `mlir-translate --mlir-to-llvmir`.
5. `llc -mtriple=spirv64-unknown-unknown -filetype=obj` (5480 bytes for the
   four-operand kernel) or `llvm-spirv` (5224 bytes) -- a real binary module,
   magic `0x07230203`, produced with no GPU and no vendor SDK. `llc` is the
   better dependency: it is the LLVM already linked, and its default output is
   SPIR-V *assembly*, which a test can FileCheck.

*The parameters are not the problem -- measured, not argued.* The kernel has 28
`OpFunctionParameter`s: per tensor, two `_ptr_CrossWorkgroup_uchar` and five
`ulong`, each its own parameter. No aggregate, no descriptor passed by value.
`runtime/vx_kernel_launch.h` and the 28-parameter fixture in
`tests/runtime/kernel_launch_test.cpp` therefore need no change, and the earlier
reading of this note -- that the four invalid `OpVariable`s were "the descriptor
slots, one per rank-2 parameter" -- was an inference from the number four, not a
measurement. `docs/gpu_backends.md` is right that the parameter shape carries
over.

*The four invalid variables are the kernel's own loop counters.* Two loops, each
with a `{vx.parallel_init}` / `{vx.parallel_bound}` store pair (outer 0..8, inner
0..4), give four `memref<i32>` slots; each becomes an `OpVariable
%_ptr_CrossWorkgroup_uint Function`, and SPIR-V refuses a `Function`-class
variable whose type points into `CrossWorkgroup`:

```
error: line 92: Storage class must match result type storage class
  %88 = OpVariable %_ptr_CrossWorkgroup_uint Function
```

Marking those four slots `#gpu.address_space<private>` before step 2 and removing
the attribute after it (the recipe upstream reports) turns them into
`%_ptr_Function_uint Function`: that error goes, and the 28 parameters are
unchanged. Confirmed here on the four-operand kernel.

*What is left, and it is wider than the slots.* The module still does not
validate. Both serializers agree on the next error:

```
error: line 127: The Object type (OpTypePointer) does not match the type that results from indexing into the Composite (OpTypePointer).
  %96 = OpCompositeInsert %_struct_18 %64 %29 0
```

`_struct_18 = OpTypeStruct %_ptr_Function_uchar %_ptr_Function_uchar %ulong`,
`%29 = OpUndef %_struct_18`, and `%64` is one of the four slot variables: a
pointer to `uint` inserted into a field typed as a pointer to `uchar`. The
unmarked module has the same shape at `%_struct_17`, with `%63`
(`_ptr_CrossWorkgroup_uint`) inserted where `_ptr_CrossWorkgroup_uchar` is
expected, after the variable error -- so the stable-looking marking swaps the
first error for the second rather than removing both.

Two kernels with no `parallel_init` at all fail as well, which is what makes this
a general blocker rather than a slot bug:

- `tests/backend/pass/spliced_block_tails.vx` -- one 1x1 tensor, 7 parameters, a
  body of two row views -- gives `%44 = OpCompositeExtract
  %_ptr_CrossWorkgroup_float %43 1`, the same message in the other direction.
- `tests/backend/pass/custom_topology_device_image.vx` -- 14 parameters -- gives
  an `OpConstantComposite` member type mismatch through `llc`, and the
  storage-class error again through `llvm-spirv`.

The pattern: every kernel that forms a memref **view** of a parameter, or keeps a
memref **value** in a slot. On this route a memref value is a five-field
descriptor struct whose pointer fields are opaque in the LLVM dialect and are
reconstructed as `i8` by the SPIR-V backend, while the pointers stored in them
are `float*` and `i32*`. Opaque pointers let that through; typed SPIR-V pointers
do not. The toolchain is not at fault: `clang -target
spirv64-unknown-unknown` on `__kernel void k(__global int *p) { p[0] = 1; }`
produces a 440-byte module `spirv-val` accepts.

*Why not the SPIR-V dialect route.* `--convert-gpu-to-spirv` gets three blockers
and loses the descriptor: it requires `spirv.entry_point_abi` on the kernel (the
spelling is `#spirv.entry_point_abi<workgroup_size = [8, 1, 1]>`); it maps
*un-annotated* memrefs to `#spirv.storage_class<CrossWorkgroup>`; it needs SCF
where Vx's kernel is a CFG (`--lift-cf-to-scf` does not lift these loops); and it
replaces the memref with an `rtarray`, dropping shape and stride.

*The XeVM route, now measured.* `mlir-opt --gpu-lower-to-xevm-pipeline` exists in
this toolchain and does run against a Vx kernel. It wraps the kernel body in
`gpu.warp_execute_on_lane_0` and then fails with `expects region #0 to have 0 or
1 blocks`: Vx's kernel body is a CFG, and the pipeline wants a body it can put in
a warp-execute region. The smallest kernel fails earlier and differently: `LLVM
Translation failed for operation: builtin.unrealized_conversion_cast` on
`!llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> ->
memref<1x1xf32, 1>`. So both routes stop on one precondition -- a device kernel
whose body is neither CFG-shaped nor memref-shaped -- for different reasons, and
neither reaches a module the validator accepts today.

*Already known, and it still holds:* the descriptor keeps `i64` sizes and strides
only with `use-64bit-index`. Without the flag they come out `i32`, and
`vx_launch_param_width` counts a rank-2 memref as two pointers plus an `i64`
offset, two `i64` sizes and two `i64` strides. Set the flag.

*The next step, and what it costs.* Not a parameter-ABI change. The device side
has to stop using memref values: either the parallel-loop counters stay in SSA
values instead of `memref<i32>` slots (removes the four variables and both errors
together), or views of a parameter are expressed without
`memref.reinterpret_cast` (which is what builds the descriptor that does not
survive). Both are Vx-side lowering questions. If neither works, the work moves
to the LLVM/SPIR-V side, where a descriptor's pointer fields must agree with the
pointers that fill them; no pass mixture or flag found so far does that, and
neither does `--use-64bit-index`.

### Slice 5 — spawn bodies checked against `dtypes:` (landed as e9fbd3b3; shared item 4)

*Change.* Landed as e9fbd3b3, at the `let` rather than as the body walk this note
planned. `check_element_type` gained the span of what it is reporting on, so the
error points at the binding; `check_element_type_in_active_region` asks the
region's own device through the region's default space; and each binding checks
every element type its type names — scalars, tensors and vectors, through the
wrappers. A body walk would have re-reported an arithmetic tree once per
subexpression; a binding is one mistake. A value that appears only as the
region's result is still unchecked, which the commit records.

*Proves.* `tests/frontend/fail/f64_scalar_inside_a_device_region.vx` — an f64
scratch value in a region on a topology whose `dtypes:` is the Arc's list, with
the machine declared inline so the fixture needs no flag — reports E6026 naming
the binding and the declared list. The same fixture with f32 compiles and runs,
f64 on the host is untouched, a topology declaring no `dtypes:` still constrains
nothing (`src/hir/check/transfer.rs:387-388`), and with the hook disabled the
fixture compiles, which is what shows the expectation is what fails. No GPU, no
SDK; `compile_test` 16 of 16 and the corpus sweep unchanged.

*Deferred.* A value that appears only as the region's result — never bound to a
name — is still unchecked. That is its own small change.

### Slice 6 — the runtime dispatch library (not started; roadmap step 4)

*Change.* A SYCL dispatch file beside `runtime/cuda_dispatch.cpp` -- the
runtime upstream settled on for this backend (docs/gpu_backends.md, "SYCL
first", decided 2026-10-05; SYCL as a *runtime*, not as a compile target). The
kernel bundle loads the SPIR-V image this plan produces, oneMKL takes the same
queue, and Level Zero is what SYCL runs on underneath. One consequence the
plan has to carry: the library is built with oneAPI's compiler, so unlike the
CUDA and Vulkan heads it cannot be a plain C++ file that builds everywhere, and
the no-SDK rule below applies to the *rest* of the tree, not to this file.
implementing the `vx_plugin_*` entry points — alloc-and-transfer
(`include/vx_hardware_runtime.h:42`), dispatch (`:342`), await, read-back,
free, control — reusing the vendor-free argument marshalling in
`runtime/vx_kernel_launch.h`. `vx_launch_build_params` (`:86`) is a pure
arithmetic claim about the memref descriptor layout and is shared by every
backend as-is; `vx_launch_entry_param_count` (`:162`) is the one piece that is
not, because it counts a signature in PTX text and SPIR-V is binary — this
slice gives that check a SPIR-V answer or drops it for this route, and says
which in the PR. An arm in `build.rs:449-454` in the shape the file already
uses: the new library is built only when the oneAPI toolchain is present, and the
host shim answers otherwise — so a build with neither SDK is exactly today's
build (`build.rs:429-433` states the rule for CUDA; the same rule applies).
`kind=matmul` routes to oneMKL (settled, §6); everything else launches the
image. The allocation side consults `vx_space_access` as §4 describes.

*Proves.* CI takes the no-SDK arm and stays green; the vendor-free pieces have
unit tests under `tests/runtime/` — the payload walk with `abi=1`/`imagebin=`,
and the marshalling against the reference signature (28 parameters for the four
rank-2 operands of `placed_kernel_four_operands.vx`) that
`kernel_launch_test.cpp` already builds. The library's own compilation and
execution happen only where the SDK exists, which is exactly CUDA's situation
in a CUDA-less CI.

*Deferred.* Running two device kinds from one program (shared item 5);
precedence when both a CUDA toolkit and a oneAPI install are present (§6
question 6); oneMKL performance work beyond correct routing.

### Slice 7 — parity and conformance (not started; shared item 6 + Tier 3 §5)

*Change.* A vendor-neutral set of placed-kernel programs, one per shape the
reference test covers, starting from
`tests/backend/pass/placed_kernel_four_operands.vx`, each runnable against the
CPU path and against the A770, marked `REQUIRES: gpu` so CI skips it the way a
`REQUIRES: ane` fixture skips on a Linux box (`docs/DEVELOPER_GUIDE.md:181-185`
lists the mechanism).

*Proves.* The roadmap's bar: same program, same numbers as the CPU path, the
runs recorded in the PR by the hardware owner. Nothing in this slice runs in
ordinary CI — that is what Tier 3 means, and it is why §5 lists it as an
obligation rather than a test.

*Deferred.* Nothing; this is the last slice.

### Order

Slices 0–3 and 5 have landed: the machine file as 886a27ec with its dtype rows as
d9822b2b, the address-space table as abd6eedf (whose emitted-MLIR diff over all
1094 fixtures cleared), the pipeline table as a7b44472, the payload version key
as c57fe069, and the dtypes check as e9fbd3b3. Slice 4 needs 1 and 2 (a gate entry and somewhere to put a binary
image); slice 5 is independent of everything; slice 6 needs 2 and 4 (the image
in the payload, and an image worth launching); slice 7 needs 6. Slices 4 and 5
can both merge before any runtime exists, which is the roadmap's
CI-testable-half-first rule.

**The slice-4 investigation was written once and lost when the development
machine's terminal was killed under memory pressure.** It has since been rerun
from scratch and its result is recorded under slice 4, so nothing is outstanding
from that loss; the shape described there is measured, not planned.

______________________________________________________________________

## 4. What the A770 measures, and where it constrains the design

Only the measured facts that change a decision; all of them are recorded with
their provenance in `fleet/arc-a770.vx`.

- **16 GiB part, 15.11 GiB usable.** The driver reports 16225243136 bytes
  (`fleet/arc-a770.vx:36-40`). `capacity:` stays 16 GiB — `fleet/README.md`'s
  rule is the hardware, not the deployment — so a working set between 15.11 and
  16 GiB is admitted here and would fail on the card. That is the open check
  shared with #285, recorded rather than folded in; it does not change any
  backend design.

- **64 KiB workgroup shared local memory through Level Zero, 48 KiB through
  Vulkan.** The hardware ceiling is 65536 bytes; Vulkan on the same card reports
  `maxComputeSharedMemorySize` 49152 (`fleet/arc-a770.vx:55-60`). The machine
  file declares the hardware's 64 KiB, and the API's smaller number is a
  backend detail: whichever runtime the backend uses, a kernel's static scratch
  must be checked against *that API's* ceiling — 64 KiB through Level Zero,
  which the SYCL runtime sits on,
  route, 48 KiB on a Vulkan route — and a tile that exceeds it must be a
  compile error, never silently shrunk. This is one concrete reason the
  roadmap's compile-error rule exists.

- **1024-invocation workgroups, preferred multiple 64**
  (`fleet/arc-a770.vx:56-57`). Launch geometry flows through the payload's
  `launch=` entry (`src/dialect/VxLowering.cpp:2209-2225`), and the flat path's
  default block width is 128 (`src/codegen/flat/emit/parallel.rs:166-168`) —
  a multiple of 64 and a divisor of 1024, so today's numbers happen to fit.
  The machine model has no field for either figure (§6 question 4), so the
  backend must respect them without the checker's help for now.

- **No f64.** `cl_khr_fp64` is absent from the OpenCL device and
  `shaderFloat64` is false under Vulkan (`fleet/arc-a770.vx:12-18`).
  `dtypes:` omits f64 (`:86`), so an f64 tensor placed there is E6026 before
  anything runs, and slice 5 extends the same error to scalars inside the
  spawn body. This is the first discrete-GPU case of the roadmap's rule that a
  device limitation is a compile error (`fleet/m4-uma.vx` was the UMA case).

- **Resizable BAR gives a 16 GiB aperture.** The card's prefetchable PCI BAR
  is the full 16 GiB (`lspci` shows `Memory at 7800000000 (64-bit,
  prefetchable) [size=16G]`), so the host *can* address the whole device
  memory. Whether the host can *read* a given allocation is then the backend's
  choice at the allocation call — `zeMemAllocDevice` hands out device memory
  the host cannot touch, a host-visible allocation answers reads. That makes
  host access a **backend policy**, not a hardware impossibility, and the
  policy has to match what the machine file declared: `fleet/arc-a770.vx`
  declares HBM and SMEM `managed: explicit`, so a program staged on this model
  expects transfers. `vx_space_access` exists to police exactly this — the
  enum at `include/vx_hardware_runtime.h:24-33` travels with every
  `vx_plugin_alloc_and_transfer` call (`:42-44`), and
  `runtime/cuda_dispatch.cpp:555-622` is the existing precedent: when the
  model claims the host may read a space the backend cannot make readable, the
  backend says so rather than staying quiet. Slice 6's design review decides
  which allocation each A770 space gets and how it answers `space_access`.

______________________________________________________________________

## 5. Tier 3 obligations, as a checklist for this backend

From the roadmap's Tier 3 definition, spelled out so a reviewer can tick them:

- [ ] **Builds with no vendor SDK installed.** The default CI build compiles
  `vxc` and runs the fixtures with no Level Zero, no oneMKL, no Vulkan present.
  The device-image half is in-tree MLIR, so nothing vendor-owned is needed to
  produce it. The dispatch library follows `build.rs`'s CUDA precedent: built
  when the SDK is found, host shim when it is not, and the no-SDK build stays
  green (slice 6).
- [ ] **Device-image half checked with FileCheck.** A fixture under
  `tests/backend/` with a `RUN: vxc ... | FileCheck` line in the shape of
  `tests/backend/pass/custom_topology_device_image.vx`, plus the image-content
  assertion beside `tests/integration_test/device_image_test.rs` (decoded bytes
  carry the SPIR-V magic). Slice 4.
- [ ] **Vendor-free runtime pieces unit-tested under `tests/runtime/`.** The
  payload walk (`abi=1`, `imagebin=` decode and malformed refusal) and the
  marshalling from `runtime/vx_kernel_launch.h` against the reference
  four-operand signature, next to the existing `kernel_launch_test.cpp`.
  Slices 2 and 6.
- [ ] **Parity runs recorded in the PR, tests marked `REQUIRES: gpu`.** The
  conformance set run on the A770 against the CPU path — same program, same
  numbers — with the runs pasted into the PR. CI never needs the card because
  the feature skips. Slice 7.
- [ ] **A named owner at merge.** The hardware owner for this card. A backend
  that loses its owner is marked unmaintained in the roadmap's status table,
  not reverted, and a Tier 3 regression never blocks `main`.

______________________________________________________________________

## 6. Settled and open

### Settled — quoting `docs/gpu_backends.md` (PR #1138)

- **"Matmul goes to a vendor library"** (cuBLAS today; oneMKL would be the
  Intel parallel). On this backend, `kind=matmul` reaches oneMKL and generated
  kernels are for everything else. A hand-tiled GEMM is out of scope by
  decision, not by omission.
- **"SYCL as a compile target is declined: Vx's checker already does the job
  SYCL's C++ layer does. What the SYCL stack offers Vx is its runtime (Level
  Zero) and its libraries (oneMKL), and those are used directly."** The route
  is therefore SPIR-V image + SYCL runtime + oneMKL as a library — upstream
  settled on the SYCL *runtime* on 2026-10-05 rather than calling Level Zero
  itself, because oneMKL takes a `sycl::queue` (docs/gpu_backends.md, "SYCL
  first") — which is what `arch: spirv64` in `fleet/arc-a770.vx:74` already
  names.
- **"A device limitation is a compile error, never a silent change."** The
  A770's missing f64 is the first discrete-GPU case: no `cl_khr_fp64`,
  `shaderFloat64` false, `dtypes:` without f64, E6026 before anything runs —
  and slice 5 closes the remaining hole, the f64 scalar inside a spawn body
  that passes the checker today.

### Open — for #1137

1. **Where the SPIR-V device image is produced.** In-tree, as a sibling of
   `deviceImageOf` (`src/dialect/VxLowering.cpp:2673`) — the roadmap's step 3,
   which slice 4 implements — or by a vendor plugin through `VxHardwarePlugin`
   (`src/plugin/hardware_trait.rs:56`; the compile-time consultation point is
   `src/codegen/lower/tensors.rs:146-149`). The investigation's blocker list is
   the input to this decision.
2. **Whether Vulkan is a second runtime over the same SPIR-V half.** The same
   image could load under Level Zero and under Vulkan with different capability
   envelopes — 64 KiB vs 48 KiB shared memory is the measured difference — or
   Vulkan could be declined the way SYCL-as-a-compile-target was declined. This
   plan does not pre-answer it.
3. **Which SPIR-V pipeline.** *Answered by the investigation: neither of the two
   the roadmap names.* `convert-gpu-to-spirv` fights three things Vx has (a CFG,
   private allocas, and a descriptor ABI it replaces with an `rtarray`), so the
   route is `--convert-gpu-to-llvm-spv=use-64bit-index` → hoist with
   `llvm.target_triple = "spirv64-unknown-unknown"` → `--convert-to-llvm
   --reconcile-unrealized-casts` → `mlir-translate --mlir-to-llvmir` → `llc
   -mtriple=spirv64-unknown-unknown -filetype=obj`. It produces a 5.5 KB SPIR-V
   module with the right magic on the first try. Recorded under slice 4, with the
   one validation error that is left and the ABI question it raises.
4. **How launch geometry is checked.** 1024 max invocations and a preferred
   multiple of 64 are API-visible facts the machine model has no field for. The
   enforcement mechanism — clamped at launch, a new declaration on `Topology`,
   or a compile error — is open; what is settled is that silently clamping a
   placement the model admitted is not allowed.
5. **When the plugin ABI freezes.** Slice 2 adds `abi=1`, but the roadmap says
   the `vx_plugin_*` interface is not frozen; what `abi=2` might carry, and
   when the freeze happens, stays with #1137.
6. **Dispatch precedence before shared item 5 lands.** One library per build
   (`build.rs:449-454`) means a machine with both a CUDA toolkit and a Level
   Zero SDK picks one; the arm must state its precedence, and moving to dispatch
   routed by topology is shared item 5's job, not this backend's.
7. **The ReBAR allocation policy.** Which A770 spaces get `zeMemAllocDevice`
   versus a host-visible allocation, and how each answers `vx_space_access` —
   a design-review item on slice 6 (§4).
8. **bf16 in `dtypes:`.** Deliberately omitted pending a citable source
   (`fleet/arc-a770.vx:82-85`); adding it is one token once a source exists.
9. **How a device kernel avoids memref values — the top item now.** Measured, and
   the opposite way round from the first reading: the parameters are fine (28
   separate `OpFunctionParameter`s, two pointers and five scalars per rank-2
   tensor, so `vx_launch_param_width` and the 28-parameter fixture do not move).
   What SPIR-V refuses is a memref *value* on the device side — a row view of a
   parameter, or a loop counter kept in a `memref<i32>` slot. The slot case has a
   fix (mark the slots private before `convert-gpu-to-llvm-spv`, and the
   storage-class error goes), but the next error is a descriptor whose pointer
   fields are typed `i8` while the pointers filling them are `float*`/`i32*`, and
   that one appears in kernels with no slots at all. So the question is where the
   device side stops using memref values, and which route (the LLVM one, or the
   XeVM one, blocked on Vx's CFG body) can carry it. Details and the failing
   instructions are in slice 4 above and on #1138.

______________________________________________________________________

## What lands next

Slices 0–3 and 5 have landed, with the machine file's dtype rows in CI beside them.
Next is slice 4, the device image, which merges on FileCheck evidence alone
starting from the investigation the lost work had begun. After that come the runtime (slice 6) and the parity runs on the
card (slice 7). A reviewer of each PR checks exactly the "proves" line written
against it above.
