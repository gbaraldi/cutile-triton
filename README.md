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

## State (August 2026, H100 NVL, triton 3.7.1, cuTile 0.3.2)

- **Coverage**: 1,810 / 1,820 of cuTile 0.3.2's own device test suite passes
  through this backend (the tail: masked `atomic_cas` has no `tt` equivalent,
  one decrementing-StepRange `LoopOp`, one triton-internal crash, and tests
  asserting native-implementation specifics).
- **Performance** vs native tileiras on the cuTile example suite: 8 of 11
  workloads at or above native (fft 3.6×, layernorm-fwd 2.06×, softmax 1.63×);
  tensor-core-scheduling-heavy kernels behind (fmha 0.80×, moe 0.55×,
  batchmatmul 0.50× — tileiras' automatic warp specialization).
- **Autotuning**: dot-kernels race {TMA, pointer}×{arg hints}×{4, 8 warps}
  once per specialization, mirroring `@triton.autotune`.
- **Hints**: `ArraySpec` maps to `tt.divisibility` argument attributes +
  compile-time-constant unit strides (→ `LDG.E.128` vectorization).
- **TMA**: eligible `PartitionView`s lower to 2–5-D tensor descriptors;
  NegInf padding is recovered via `select(mask, tma_load, -inf)`.

## Layout

| Path | Purpose |
|---|---|
| `src/TileTriton.jl` | package module: `triton_kernel`/`launch!`, `code_triton` reflection, `install_shim!`, `set_target!`/`use_rocm!` |
| `src/emitter.jl` | `TritonEmitter`: StructuredIRCode → TTIR walker (~1,900 lines, the core) |
| `src/runtime.jl` | `TritonRun`: in-process compile (PythonCall) + launcher, vendor-parameterized target |
| `src/shim.jl` | `TritonShim` + `install_shim!()`: overwrite `cuTile.cufunction` so unmodified cuTile code runs here (opt-in) |
| `src/dialects/Triton.jl` | generated `tt` dialect builders (from Reactant.jl, tblgen'd from Triton's `.td`) |
| `ext/TileTritonAMDGPUExt.jl` | **experimental** ROCm target (see below) |
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

## AMDGPU (experimental, untested on hardware)

The compile side (TTIR → hsaco via the wheel's `hip` backend) is known-good
from offline tests; the load/launch side needs validation on a ROCm box:

```julia
using AMDGPU, TileTriton
TileTriton.use_rocm!()        # gfx arch + wavefront size from the device
k = TritonRun.triton_kernel(my_cutile_kernel, argtypes; name="k")
TritonRun.launch!(k, grid, rocarrays...)
```

Things to watch when bringing it up: the launcher passes raw `Ptr` kernel
params via `hipModuleLaunchKernel` (mirrors the CUDA driver ABI); triton's
hip metadata (`shared`, `global_scratch_size`) is honored the same way; the
TMA/descriptor path and the tf32-vs-hints autotune splits are NVIDIA-tuned —
expect the pointer path everywhere on CDNA, and `warp_size=64` changes the
threads-per-CTA arithmetic (handled via `TargetInfo`).

## Notes

- `Triton.jl` is generated code (mlir-jl-tblgen over Triton's tablegen
  definitions, via Reactant.jl's build); Triton is MIT-licensed, Reactant.jl
  MIT.
- `TritonEmitter.jl` carries a method overwrite fixing a cuTile 0.3.2
  front-end bug (`promote_scalar_type` vs `Union` types on Julia 1.12) —
  candidate upstream PR.
- Known upstream findings worth filing: `tt.dot` int8 with K<32 silently
  miscompiles (2×) via `IRSource`; tf32 descriptor matmuls are slow on
  sm_90 (no upstream tutorial even exercises them).

Prototype quality: this exists to inform the design of a real Triton target
inside cuTile.jl (behind `emit_intrinsic!`), not to be a package.
