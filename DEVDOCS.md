# TileTriton developer notes

How the backend is built, the decisions behind it, and the workarounds it
carries for its dependencies. User-facing material is in the README.

## Layout

| Path | Purpose |
|---|---|
| `src/TileTriton.jl` | package module and exports |
| `src/emitter.jl` | `TritonEmitter`: StructuredIRCode → TTIR walker (the core) |
| `src/runtime.jl` | `TritonRun`: in-process compile, candidates, launch, launch-time checks, reflection, vendor targets |
| `src/shim.jl` | `TritonShim` + `install_shim!()`: overwrites `cuTile.cufunction` and `cuTile.launch` (opt-in piracy) |
| `src/dialects/Triton.jl` | generated `tt` dialect builders (mlir-jl-tblgen over Triton's `.td`, via Reactant.jl's build; Triton and Reactant.jl are MIT) |
| `ext/TileTritonAMDGPUExt.jl` | ROCm target: HIP load/launch, ROCArray `TileArray`s, cuTile's host RNG on ROCArrays |
| `test/runtests.jl` | standalone correctness suite, both vendors |
| `test/cutile_suite.jl` | cuTile's own device suite through the shim, with a CUDA→AMDGPU compat layer |
| `bench/` | example benchmarks (`CUTILE_BACKEND=native\|triton bench/examples.jl`), micro-benchmarks, ROCm comparisons, SpMM study |
| `research/` | report, lowering dumps, SASS scheduling analysis, logs |

## Pipeline

`emit_ttir(f, argtypes)` runs cuTile's front end (`code_structured`, keeping
the `@compiler_options` metadata it normally drops), walks the structured IR
and builds TTIR with MLIR.jl's generic builders; the `tt` dialect is never
registered, so the module is printed as generic-form text. It returns the
text, the flattened argument spec and metadata (whether the kernel has a dot
or atomics, the in-tile spans, the kernel's hints).

`TritonRun` compiles that text in-process with the pinned wheel
(`triton.compiler.compile` on an IRSource) and loads the binary with CUDA.jl
or HIP. `compile` starts an IR file after that file's own stage, so a `.ttir`
input would skip `make_ttir` (inliner, canonicalize, combine, CSE, loop
unroll, and before sm_90 the descriptor-to-pointer rewrite, without which
every tensor-descriptor kernel failed there); `TritonRun` runs that stage
first, as for a Python kernel. It costs ~2% of compile time and changes no
measured kernel time, since cuTile's optimizer already simplifies. The kernel ABI is triton's: the declared arguments (a TileArray
flattens to pointer, sizes, strides), the implicit KernelState seed, then a
global-scratch and a profile-scratch pointer. When a kernel reports
`global_scratch_size` (TMA descriptors on NVIDIA; AMD kernels too from triton
3.8) the launchers allocate grid volume × that size, as triton's do; the
profile slot is NULL.

Kernel-creation-time switches (`USE_TMA`, `WIDE_OFFSETS`) are module-level
`Ref`s set by `emit_ttir`'s keywords.

## Candidates and racing

`triton_kernel_candidates` builds a 4- and an 8-warp kernel (dot kernels:
{TMA, pointer} × {argument hints} × warps, deduplicated by TTIR). The shim
times each once on the first launch of a specialization and keeps the
winner, like `@triton.autotune`. Every TileArray argument's memory extent is
snapshotted beforehand and restored after each candidate, so read-modify-write
kernels (accumulation, atomics, spin locks) see exactly one application;
beyond 2 GiB of arguments it skips the race and takes the first candidate.
On NVIDIA, pipeline stages drop until a kernel fits the device's opt-in
shared memory (AMD has no such query; triton defaults to 2 stages there).

## Hints, TMA and launch hints

- `ArraySpec` alignment and divisibility become `tt.divisibility` argument
  attributes, and a contiguous first dimension becomes a constant 1 stride,
  which lets AxisInfo prove contiguity (→ 128-bit vector loads).
- Eligible partition views (2–5-D, contiguous inner dim, box ≤ 256 per dim,
  ≥ 16-byte inner box, sizes that fit i32, elements of at most 16 bits) lower
  to tensor descriptors: TMA on sm_90+, plain loads in triton's lowering
  before that, where 16-bit floats fail and are kept off descriptors. tf32
  descriptor matmuls are a regression on triton 3.7's Hopper pipeline, hence
  the size gate. Descriptors zero-fill, so any other padding is recovered
  with a select against the bounds mask.
- Launch hints resolve as in cuTile (explicit value, else `@compiler_options`
  for the target; validated by cuTile's `validate_hint`). `num_worker_warps`
  pins the warp count, `occupancy` becomes `maxnreg` (NVIDIA, 64K registers
  per SM) or `waves_per_eu` (AMD, 4 SIMDs per CU), and `num_ctas > 1` errors
  before sm_100 or on AMD like the native backend, then is unused.

## Index width

- Loops take the range's integer width; tile counts divide at the size's
  width and narrow to cuTile's Int32 index space.
- `view_ptrs_mask` adds each tile's origin to the base pointer as one scalar
  i64 and keeps in-tile offsets at the view's index width, i32 for
  Int32-indexed arrays. All-i64 offsets cost 22% on an H100 row reduction; the
  split form keeps AMD's canonical splat(base) + i32 offsets (buffer ops).
- An in-tile span can still exceed 2³¹ elements on an Int32-indexed array
  (a 16×64 tile of a 34.6M×64 array). The emitter records a `SpanTerm` per
  i32-offset view (tile extent, step, and which ABI slots hold its sizes and
  strides); `launch!` and the shim evaluate the largest in-tile offset and
  the last tile's end from the actual arguments and run a lazily compiled
  `wide_offsets=true` build when either passes 2³¹ − 1. This is
  JuliaGPU/CUDA.jl#3389's launch-checked specialization; the check costs
  nothing measurable per launch.
- Gather/scatter views compute offsets at i64: their sparse coordinates are
  data no launch check can bound.
- Missing trailing tile indices are 0, as natively, so an out-of-range index
  selects no element instead of wrapping into another tile.

## Control flow and atomics

- `LoopOp` (decrementing ranges, `break`/`continue`) becomes an `scf.while`
  whose before region runs the body and ends in `scf.condition(continuing,
  values)`; an `if` that exits in a branch yields `(flag, values...)` and runs
  the rest of the body in each branch that falls through.
- Runtime throws (cuTile's `ThrowNode.runtime`) and asserts lower to
  `tt.assert` on a 1024-lane tensor whose element 0 carries the condition, so
  one thread per program reports: with every thread asserting, the CUDA
  driver prints none of the messages.
- `tt.atomic_cas` has no mask. Masked-out lanes compare-and-swap 0 for 0 at
  the largest active address (cannot change memory) and return `expected`, as
  cuTile specifies; with no active lane the CAS is skipped. A constant-false
  mask on any atomic touches no memory.

## AMD

- `use_rocm!()` sets the hip target from the device, plugs HIP module
  load/launch/sync and device copies into `TritonRun`, and retargets cuTile:
  `device_pointer` accepts ROCm device memory (by type for `AnyROCArray`, by
  a HIP memory-type query for other wrappers), and the allocating host RNG
  calls build ROCArrays.
- Every hip compile sets `AMDGCN_USE_BUFFER_ATOMICS=0` (see below); buffer
  loads and stores stay on.
- Tensor floor/ceil divisions are expanded (see below).

## Reflection

`code_triton(f, argtypes; stage, target)` emits and compiles exactly as a
launch would (hints included) and prints a stage: `:ttir`, `:ttgir`,
`:llir`, then `:ptx`/`:sass` on NVIDIA or `:amdgcn` on AMD.
`TritonRun.target(:amd; arch)` / `target(:nvidia; arch)` build a target
without its GPU (wave64 on CDNA, wave32 on RDNA). Triton uses no upstream
MLIR `gpu` dialect: TTIR (`tt`) → TTGIR (`ttg` + `amdg` / `ttng`, `nvg`) →
LLVM dialect + `rocdl` / `nvvm` → LLVM IR; NVIDIA atomics are inline PTX.

## Test harness

`test/cutile_suite.jl` includes each of cuTile's device test files in a
fresh module. `@filecheck` blocks run with `@check` neutralized (they check
native Tile IR). On ROCm a `CUDACompat` module stands in for CUDA.jl: array
constructors forward to AMDGPU, `@cuda` is CUDA.jl's (routed through the
shim), `@device_code_sass` prints AMDGCN, and `capability(device())`
answers v9.0 so both vendors take the H100's branches.

Environment overrides: `TRITON_NUM_WARPS` (skip the warp race),
`TRITON_ARG_ATTRS=0` / `TRITON_CONST_STRIDE=0` (drop hint emission),
`TRITON_DUMP_TTIR=dir`, `TRITON_KERNEL_INFO=1` (registers and spills per
kernel), `TILETRITON_MAKE_TTIR=0` (skip triton's TTIR stage),
`TILETRITON_TEST_BACKEND=rocm` (tests on AMD).

## Performance notes

H100 NVL vs native tileiras on cuTile's examples (`bench/examples.jl`,
October 2026, cuTile 1.0.0, triton 3.8.0; minimum of 20 runs, better of two
passes):

| workload | native | triton | triton / native |
|---|---|---|---|
| fft | 188 µs | 63 µs | 2.98× |
| layernorm fwd | 1,783 GB/s | 4,505 GB/s | 2.53× |
| layernorm bwd | 398 GB/s | 793 GB/s | 1.99× |
| softmax (TMA) | 1,645 GB/s | 2,740 GB/s | 1.67× |
| softmax (chunked) | 3,944 GB/s | 4,978 GB/s | 1.26× |
| matmul (f32) | 64.6 TFLOPS | 65.0 TFLOPS | 1.01× |
| transpose | 3,248 GB/s | 3,238 GB/s | 1.00× |
| vadd | 3,551 GB/s | 3,559 GB/s | 1.00× |
| moe | 90.7 TFLOPS | 88.9 TFLOPS | 0.98× |
| fmha | 294 TFLOPS | 265 TFLOPS | 0.90× |
| batchmatmul | 111 TFLOPS | 52.6 TFLOPS | 0.47× |

Against triton 3.7.1, 3.8.0 lifts moe from 52 to 89 TFLOPS and costs ~5% on
fmha, layernorm bwd and fft. batchmatmul (and fmha) are tensor-core
scheduling bound, where tileiras warp-specializes automatically; triton's
Hopper warp specialization is still immature upstream. f16 matmul 4096³
through descriptors reached 541 TFLOPS at 128×256×64 (tileiras 447, cuBLAS
512; August 2026).

MI300A vs native AMDGPU.jl (`bench/rocm_compare.jl`, October 2026, ROCm
7.2.4, cuTile 1.0.0, triton 3.8.0, untuned single tile shape):

| workload | triton | native AMDGPU.jl |
|---|---|---|
| vadd 2²⁸ | 3,024 GB/s | 2,855 GB/s broadcast / 2,795 `@roc` |
| rowsum 131072×512 | 1,839 GB/s | 191 GB/s (`sum!`, GPUArrays mapreduce) |
| softmax 4096×16384 | 2,673 GB/s | 211 GB/s (GPUArrays composition) |
| matmul f16 4096³ | 226.9 TFLOPS | 3.3 `mul!` (earlier run: 224.1 `gemm_ex`, 71.8 `gemm!`) |
| matmul f32 4096³ | 63.2 TFLOPS | 54.1 rocBLAS |

vadd is ahead of native, tile reductions and fusions are 10–13× ahead of
GPUArrays, f32 matmul beats rocBLAS by 17%, and f16 matmul matches rocBLAS's
tuned `gemm_ex` path. The benchmark builds its argument types from the actual
arrays (`ct.TileArray`), as `ct.launch` does: the hand-written `ArraySpec`s it
used before claimed no size or stride divisibility under cuTile 1.0's spec
layout, which kept loads scalar (f16 matmul was 133 TFLOPS). rowsum measured
2,441 GB/s in August; the August code and cuTile 0.3.2 rerun today on the
same kind of node measure ~1,830, as current code does, so that difference
is the node. Matmul tile shapes are NVIDIA-tuned; CDNA autotuning
(`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`) is the open lever. `mul!`
on Float16 falls into GPUArrays' generic kernel on Julia 1.12
(`generic_matmatmul_wrapper!`), an AMDGPU.jl bug.

## Workarounds and upstream findings

triton 3.8.0:

- AMD buffer-atomics lowering emits the nonexistent
  `llvm.amdgcn.raw.ptr.buffer.atomic.exch` (the real one is `*.swap`),
  breaking `atomic_xchg`; fixed by triton-lang/triton#11019 (release/3.9.x,
  not 3.8.0), and #11081 fixes swapped acquire/release fences on buffer
  atomics there too.
- AMD leaves tensor `arith.floordivsi`/`ceildivsi`/`ceildivui` unlowered
  (`unrealized_conversion_cast` at LLVM translation; still open upstream:
  NVIDIA runs `populateCeilFloorDivExpandOpsPatterns`, AMD does not).
- `tt.dot` on int8 with K < 32 silently computes 2× through IRSource (the
  Python frontend asserts K ≥ 32); int8 mma is emulated with an exact f32 dot.
- tf32 descriptor matmuls are slower than pointer loads on sm_90.

Fixed by moving from 3.7.1 to 3.8.0: AMD ran a tensor `atomic_cas` that
missed the buffer-atomic path in every thread holding a copy of an element
(later copies returned the first copy's update; triton-lang/triton#9605). 3.8
also changed the tensor descriptor type syntax (`!tt.tensordesc<64x64xf16>`,
#9851) and reports `global_scratch_size` on AMD, which the launchers now
allocate for both vendors. The xchg fix, #11081 and dedicated
`atomic_load`/`atomic_store` ops are in release/3.9.x (untagged); AMD
floor/ceil division, int8 dot K < 32 (no verifier check) and `atomic_cas`
masks are still open upstream.

cuTile.jl 1.0.0, both overridden in `src/emitter.jl`:

- `promote_scalar_type` takes `T.parameters` of a `Union` or `UnionAll` of
  tuples that passes its `<: Tuple` check (FieldError). hvcat reached it on
  cuTile 0.3.2; no kernel does on 1.0.0. Upstream: JuliaGPU/cuTile.jl#336.
- The optimizer's one in-place rewrite rule corrupts an index tile shared by
  a mask comparison and a gather; the emitter demotes it to out-of-place.
  Upstream removed in-place rewrites (JuliaGPU/cuTile.jl#306, after 1.0.0).
- A kernel with a type-unstable tuple (`flag ? (Int32(1), Int32(2)) :
  (1f0, 2f0)`) fails natively with "SSAValue %N not found in context" on
  1.0.0 and main, rather than with a clear error.

cuTile.jl's `launch` evaluates `stream=CUDACore.stream()` eagerly and its
host RNG API and `TileArray` construction assume CUDA memory; the shim and the
AMDGPU extension replace those on ROCm.
