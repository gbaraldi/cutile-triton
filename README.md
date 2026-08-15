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

| File | Purpose |
|---|---|
| `TritonEmitter.jl` | StructuredIRCode → TTIR walker (~1,800 lines, the core) |
| `TritonRun.jl` | in-process compile (PythonCall) + CUDA.jl launcher + `code_triton` reflection (`:ttir/:ttgir/:llir/:ptx/:sass`) |
| `Triton.jl` | generated `tt` dialect builders (from Reactant.jl, tblgen'd from Triton's own `.td`) |
| `triton_shim_module.jl` | overwrites `cuTile.cufunction` so unmodified cuTile code runs on this backend |
| `run_cutile_tests.jl` | runs cuTile's own device test suite through the shim |
| `test_correct.jl` | standalone correctness suite (vadd/rowsum/matmul/softmax) |
| `bench_examples.jl` | benchmarks all cuTile examples (`CUTILE_BACKEND=native\|triton`) |
| `bench*.jl`, `sweep_matmul.jl` | micro-benchmarks (TMA, fp16, descriptor overhead) |
| `stages/` | full 6-stage lowering dump of one kernel (SCI→TTIR→TTGIR→LLIR→PTX→SASS) |
| `sass/` | disassembly evidence for the tileiras-vs-triton scheduling analysis |
| `report.html` | the full investigation report (pipeline, benchmarks, SASS anatomy, upstream survey) |

## Setup

```julia
] instantiate            # Julia deps (Project.toml/Manifest.toml)
using CondaPkg; CondaPkg.resolve()   # pinned triton==3.7.1 (CondaPkg.toml)
include("test_correct.jl")
```

Requires an NVIDIA GPU (tested sm_90) and CUDA driver. `env TRITON_NUM_WARPS`,
`TRITON_ARG_ATTRS`, `TRITON_CONST_STRIDE`, `TRITON_DUMP_TTIR` override the
autotuner/hints for experiments.

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
