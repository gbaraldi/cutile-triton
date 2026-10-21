# cuTile.jl → Triton backend prototype

A prototype backend that retargets [cuTile.jl](https://github.com/JuliaGPU/cuTile.jl)
from NVIDIA Tile IR (`tileiras`) to [Triton](https://github.com/triton-lang/triton),
for NVIDIA and AMD GPUs, without Reactant and without any Triton-specific
native library:

```
Julia kernel  →  cuTile front end      (code_structured, unchanged)
              →  TritonEmitter.jl      (MLIR.jl, unregistered tt dialect → TTIR text)
              →  triton pip wheel      (in-process via PythonCall, CondaPkg-pinned 3.8.0)
              →  CUDA.jl / AMDGPU.jl   (module load + launch)
```

Kernels are written in the plain cuTile language; nothing kernel-side knows
Triton exists. Prototype quality: this exists to inform the design of a real
Triton target inside cuTile.jl, not to be a package. Internals, workarounds
and upstream findings are in [DEVDOCS.md](DEVDOCS.md).

## Status (October 2026, triton 3.8.0, cuTile 1.0.0)

| | H100 NVL (sm_90) | MI300A (gfx942, ROCm 7.2.4) |
|---|---|---|
| cuTile's device suite (pass / fail / error) | 1,916 / 0 / 0 | 1,915 / 1 / 0 ¹ |
| standalone suite (`test/runtests.jl`) | 61 / 61 | 61 / 61 |

¹ core.jl's failing-assertion test runs native cuTile on CUDA in a child
process; the standalone suite checks the same behaviour through this backend.

The whole cuTile 1.0 language is covered, including gather/scatter views,
atomics through views and masked `atomic_cas`, RNG, asserts and runtime
throws, general loops, and launch hints. On H100, 8 of cuTile's 11 example
workloads run at or above native tileiras (fft 3.0×, layernorm 2–2.5×,
softmax 1.3–1.7×), MoE at 0.98× and fmha at 0.90×; batched matmul is about
half as fast, where tileiras warp-specializes automatically. On MI300A,
against AMDGPU.jl 2.9.1: softmax 4.5× and rowsum 1.3× faster than its
GPUArrays paths, f32 matmul 1.12× rocBLAS; f16 matmul is at 0.75× rocBLAS
and vadd at 0.94× a native kernel.

## Usage

```julia
using TileTriton
k = TritonRun.triton_kernel(my_cutile_kernel, argtypes; name="k")
TritonRun.launch!(k, grid, arrays...)

TileTriton.install_shim!()     # or route all of cuTile (ct.launch, @cuda backend=cuTile) here
```

On AMD, load AMDGPU and call `TileTriton.use_rocm!()` first; the shim then
runs unmodified cuTile code on ROCArrays.

Reflection works for either vendor from any machine:

```julia
TritonRun.code_triton(f, argtypes; stage=:ttgir)    # :ttir :ttgir :llir :ptx :sass :amdgcn
TritonRun.code_triton(f, argtypes; stage=:amdgcn, target=TritonRun.target(:amd))
```

## Setup and tests

```julia
] activate .; instantiate
using CondaPkg; CondaPkg.resolve()     # pinned triton==3.8.0 (CondaPkg.toml)
```

```
julia --project=. test/runtests.jl                     # standalone suite
julia --project=. test/cutile_suite.jl [files...]      # cuTile's device suite through the shim
TILETRITON_TEST_BACKEND=rocm julia --project=. test/runtests.jl   # either one, on AMD
```
