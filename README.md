# cuTile.jl → Triton backend prototype

A working prototype backend that retargets [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl)
from NVIDIA Tile IR (`tileiras`) to [Triton](https://github.com/triton-lang/triton),
without Reactant and without any Triton-specific native library:

```
Julia kernel  →  cuTile front end (code_structured, unchanged)
              →  TritonEmitter.jl   (MLIR.jl, unregistered tt dialect → TTIR text)
              →  triton pip wheel   (in-process via PythonCall, CondaPkg-pinned)
              →  CUDA.jl            (cuModuleLoadData + cudacall)
```

Kernels are written in the plain cuTile language; nothing kernel-side knows
Triton exists.

## State (October 2026, H100 NVL, triton 3.7.1, cuTile 1.0.0)

- **Coverage**: cuTile 1.0.0's own device test suite passes in full through
  this backend: 1,916 pass / 0 fail / 0 error, including gather/scatter
  views, atomic reductions through views, masked `atomic_cas` (emulated:
  `tt.atomic_cas` has no mask), general loops (`LoopOp`) and runtime throws.
  Launch hints are honored: `num_worker_warps` sets the warp count,
  `occupancy` a register (NVIDIA) or waves-per-EU (AMD) budget, and
  `num_ctas > 1` is rejected without thread-block clusters, as natively.
- **Performance** (August 2026, cuTile 0.3.2) vs native tileiras on the cuTile
  example suite: 8 of 11
  workloads at or above native (fft 3.6×, layernorm-fwd 2.06×, softmax 1.63×);
  tensor-core-scheduling-heavy kernels behind (fmha 0.80×, moe 0.55×,
  batchmatmul 0.50× — tileiras' automatic warp specialization).
- **Autotuning**: every kernel races 4- and 8-warp builds once per
  specialization, mirroring `@triton.autotune`; dot kernels additionally race
  {TMA, pointer}×{arg hints}. Argument memory is snapshotted and restored
  around the race, so read-modify-write kernels (accumulation, atomics) see
  exactly one application. Pipeline stages drop until a kernel fits the
  device's shared memory.
- **Hints**: `ArraySpec` maps to `tt.divisibility` argument attributes +
  compile-time-constant unit strides (→ `LDG.E.128` vectorization).
- **TMA**: eligible `PartitionView`s lower to 2–5-D tensor descriptors;
  NegInf padding is recovered via `select(mask, tma_load, -inf)`.
- **Index width**: loops run at the range's integer width, and each tile's
  origin is added to the base pointer at 64 bits, so arrays beyond 2³¹
  elements address correctly while in-tile offsets stay i32. A tile can
  still span more than 2³¹ elements (e.g. 16×64 over a 34.6M×64 array), so,
  like CUDA.jl's index narrowing (JuliaGPU/CUDA.jl#3389), each launch checks
  the actual sizes and strides and runs an i64-offset build when they need it.

## Layout

| Path | Purpose |
|---|---|
| `src/TileTriton.jl` | package module: `triton_kernel`/`launch!`, `code_triton` reflection, `install_shim!`, `set_target!`/`use_rocm!` |
| `src/emitter.jl` | `TritonEmitter`: StructuredIRCode → TTIR walker (~1,900 lines, the core) |
| `src/runtime.jl` | `TritonRun`: in-process compile (PythonCall) + launcher, vendor-parameterized target |
| `src/shim.jl` | `TritonShim` + `install_shim!()`: overwrite `cuTile.cufunction` and `cuTile.launch` so unmodified cuTile code runs here (opt-in) |
| `src/dialects/Triton.jl` | generated `tt` dialect builders (from Reactant.jl, tblgen'd from Triton's `.td`) |
| `ext/TileTritonAMDGPUExt.jl` | ROCm target (validated on MI300A, see below) |
| `test/runtests.jl` | standalone correctness suite |
| `test/cutile_suite.jl` | runs cuTile's own device test suite through the shim |
| `bench/` | example benchmarks (`CUTILE_BACKEND=native\|triton bench/examples.jl`) + micro-benchmarks |
| `research/` | report.html, 6-stage lowering dumps, SASS scheduling analysis, logs |

## Setup

```julia
] activate .; instantiate      # Julia deps
using CondaPkg; CondaPkg.resolve()   # pinned triton==3.7.1 (CondaPkg.toml)
using TileTriton
include("test/runtests.jl")
```

Requires an NVIDIA GPU (tested sm_90). `TRITON_NUM_WARPS`, `TRITON_ARG_ATTRS`,
`TRITON_CONST_STRIDE`, `TRITON_DUMP_TTIR` env vars override the autotuner and
hint emission for experiments.

## AMDGPU (validated on MI300A, gfx942; ROCm 6.4.2 and 7.2.4)

The same kernels compile and run through the wheel's `hip` backend:

```julia
using AMDGPU, TileTriton
TileTriton.use_rocm!()        # gfx arch + wavefront size from the device
k = TritonRun.triton_kernel(my_cutile_kernel, argtypes; name="k")
TritonRun.launch!(k, grid, rocarrays...)
```

Validated on an MI300A (ROCm 7.2.4, cuTile 1.0.0; earlier rounds also under
ROCm 6.4.2):

- standalone suite 43/43 (`TILETRITON_TEST_BACKEND=rocm julia --project=.
  test/runtests.jl`), including the tf32 matmuls — CDNA3's XF32 MFMA
  absorbs `inputPrecision=tf32` at tf32-like accuracy — and the >2³¹-element
  addressing tests;
- **cuTile's own device suite: 1,915 pass / 1 fail / 0 error** through the
  shim, file for file the CUDA run except one test: core.jl's failing
  assertion runs native cuTile on CUDA in a child process (the standalone
  suite checks the same through this backend, on both vendors). The compat
  layer in `test/cutile_suite.jl` maps the suite's CUDA calls to AMDGPU
  (`@device_code_sass` shows AMDGCN; `capability` answers like the H100 so
  both runs take the same branches); the extension gives cuTile's host RNG
  API and `TileArray` construction (views, wrappers) their ROCArray
  equivalents;
- head-to-head vs native AMDGPU.jl on the same workloads
  (`bench/rocm_compare.jl`, ROCm 7.2.4, untuned single tile shape):

  | workload | triton | native AMDGPU.jl |
  |---|---|---|
  | vadd 2²⁸ | 2,658 GB/s | 2,988 GB/s broadcast / 2,881 `@roc` |
  | rowsum 131072×512 | 2,441 GB/s | 192 GB/s (`sum!`, GPUArrays mapreduce) |
  | softmax 4096×16384 | 2,660 GB/s | 223 GB/s (GPUArrays composition) |
  | matmul f16 4096³ | 137.6 TFLOPS | 224.1 `gemm_ex` / 71.8 `gemm!` / **3.0 `mul!`** |
  | matmul f32 4096³ | 40.0 TFLOPS | 53.8 rocBLAS |

  Streaming is ~10% behind native; tile-level reductions/fusions are >10×
  ahead of GPUArrays; matmul sits between rocBLAS's legacy and tuned paths
  (NVIDIA-tuned tile shapes — CDNA autotuning is the open lever). The f16
  native column is a ladder: `gemm_ex` with f32 compute is the
  Tensile-tuned path; `gemm!` maps to legacy untuned `hgemm`; and `mul!`
  is an AMDGPU.jl bug on Julia 1.12 (`generic_matmatmul_wrapper!` sends
  Float16 to GPUArrays' generic kernel — 75× off the fix, worth an
  AMDGPU.jl issue). Numbers identical under prerelease rocm/7.14.0.

Bring-up notes: the AMD launcher ABI matches CUDA's (declared args + a
NULL global-scratch slot + profile scratch; `global_scratch_size` is
absent from hip `KernelMetadata`), `warp_size=64` flows through
`TargetInfo`, `num_stages` defaults to 2 (64KB LDS), and the TMA path is
NVIDIA-only so `use_rocm!()` runs pointer loads everywhere.
Every hip compile (launches and reflection alike) sets
`AMDGCN_USE_BUFFER_ATOMICS=0`: triton 3.7.1's
buffer-atomics lowering emits the nonexistent LLVM intrinsic
`...raw.ptr.buffer.atomic.exch` (the real one is `*.swap`), killing every
`atomic_xchg` at compile — worth an upstream issue.
The AMD backend also leaves tensor `arith.floordivsi`/`ceildivsi`/
`ceildivui` unlowered (NVIDIA lowers them), so the emitter expands those to a
truncating divide plus a correction. And triton 3.7.1 runs an AMD tensor
`atomic_cas` that misses the buffer-atomic path in every thread holding a copy
of an element (fixed upstream in triton-lang/triton#11841, unreleased), so on
AMD the emitter spreads each tensor CAS under 8,192 elements over at least
1,024 lanes, giving every element one owner.

Reflection works for either vendor from any machine:
`TritonRun.code_triton(f, argtypes; stage=:ttgir|:llir|:amdgcn,
target=TritonRun.target(:amd))` compiles for gfx942 without an AMD GPU
(`target(:nvidia)` with `:ptx`/`:sass` the other way).

## Notes

- `Triton.jl` is generated code (mlir-jl-tblgen over Triton's tablegen
  definitions, via Reactant.jl's build); Triton is MIT-licensed, Reactant.jl
  MIT.
- `TritonEmitter.jl` carries two cuTile workarounds, both candidate upstream
  PRs: a method overwrite for a front-end bug (`promote_scalar_type` vs `Union`
  types on Julia 1.12, still present in 1.0.0), and a demotion of the
  optimizer's one in-place rewrite rule, which corrupts index tiles shared by
  a mask comparison and a gather.
- Known upstream findings worth filing: `tt.dot` int8 with K<32 silently
  miscompiles (2×) via `IRSource`; tf32 descriptor matmuls are slow on
  sm_90 (no upstream tutorial even exercises them).

Prototype quality: this exists to inform the design of a real Triton target
inside cuTile.jl (behind `emit_intrinsic!`), not to be a package.
