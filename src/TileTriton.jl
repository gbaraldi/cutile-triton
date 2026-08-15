"""
    TileTriton

Prototype Triton backend for cuTile.jl: Julia tile kernels → cuTile front end
(`code_structured`) → TTIR (MLIR.jl, unregistered `tt` dialect) → the
CondaPkg-pinned `triton` wheel (in-process via PythonCall) → CUDA.jl (or,
experimentally, AMDGPU.jl through the package extension).

Entry points:
- `TritonRun.triton_kernel(f, argtypes; name)` / `TritonRun.launch!` —
  compile and launch a cuTile kernel through Triton.
- `code_triton(f, argtypes; stage=:ttir|:ttgir|:llir|:ptx|:sass)` — reflection.
- `install_shim!()` — route *all* cuTile compilation (`@cuda backend=cuTile`,
  `ct.launch`) through this backend, so unmodified cuTile code runs on it.
- `set_target!(backend, arch, warp_size)` — override the compile target
  (defaults to the active CUDA device; the AMDGPU extension provides
  `use_rocm!()`).
"""
module TileTriton

include("emitter.jl")     # module TritonEmitter
include("runtime.jl")     # module TritonRun
include("shim.jl")        # module TritonShim + install_shim!

using .TritonEmitter: emit_ttir, ArgSpec
using .TritonRun: triton_kernel, launch!, code_triton, set_target!

"""
    use_rocm!()

Switch compilation and launching to the active AMD GPU (requires `using
AMDGPU`; provided by the package extension). Experimental.
"""
function use_rocm! end

export TritonEmitter, TritonRun, TritonShim, use_rocm!,
       emit_ttir, triton_kernel, launch!, code_triton, set_target!,
       install_shim!

end # module
