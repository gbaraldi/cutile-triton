# Validated on MI300A (gfx942, ROCm 6.4.2): full standalone suite passes.
# The HIP launch ABI mirrors the CUDA driver one (array of pointers to the
# argument values; triton's hip metadata provides warp_size/shared).
module TileTritonAMDGPUExt

using AMDGPU
using TileTriton
using TileTriton.TritonRun: TritonRun, set_target!, _load_module, _raw_launch, _sync
import cuTile

# cuTile's generic TileArray(arr) refuses plain-`Ptr` arrays as a host-memory
# guard; ROCArray hands out plain `Ptr` for device memory (HIP unified
# addressing), so identify it explicitly. This is what lets unmodified cuTile
# launches (`cuTile.launch`, the test suite) run on ROCm through the shim.
cuTile.device_pointer(arr::ROCArray{T}) where {T} = pointer(arr)

"""
    TileTriton.use_rocm!()

Point the backend at the active AMD GPU: compile TTIR with the triton wheel's
`hip` backend (gfx arch + wavefront size from AMDGPU.jl) and load/launch the
resulting hsaco through HIP.
"""
function TileTriton.use_rocm!()
    dev = AMDGPU.device()
    # "gfx942:sramecc+:xnack-" → "gfx942"
    arch = String(first(split(AMDGPU.HIP.gcn_arch(dev), ':')))
    ws = Int(AMDGPU.HIP.wavefrontsize(dev))
    set_target!("hip", arch, ws)
    _load_module[] = _hip_load
    _raw_launch[] = _hip_launch
    _sync[] = AMDGPU.synchronize
    # triton 3.7.1's buffer-atomics lowering emits the nonexistent LLVM
    # intrinsic ...raw.ptr.buffer.atomic.exch (the real one is *.swap), which
    # kills every atomic_xchg at compile. Buffer loads/stores stay on.
    TritonRun._py_setenv!("AMDGCN_USE_BUFFER_ATOMICS", "0")
    @info "TileTriton targeting ROCm" arch wavefront=ws
    return nothing
end

function _hip_check(status)
    status == AMDGPU.HIP.hipSuccess ||
        error("HIP error: $(unsafe_string(AMDGPU.HIP.hipGetErrorString(status)))")
end

function _hip_load(bin::Vector{UInt8}, name::String, shared::Int)
    mod = Ref{AMDGPU.HIP.hipModule_t}()
    _hip_check(AMDGPU.HIP.hipModuleLoadData(mod, bin))
    fun = Ref{AMDGPU.HIP.hipFunction_t}()
    _hip_check(AMDGPU.HIP.hipModuleGetFunction(fun, mod[], name))
    return mod[], fun[]
end

function _hip_launch(fun, types, vals; threads, blocks, shmem)
    # pack kernel params CUDA-driver-style: an array of pointers to the
    # argument values
    refs = Any[]
    ptrs = Ptr{Cvoid}[]
    for (T, v) in zip(types, vals)
        r = Base.RefValue{T}(convert(T, v))
        push!(refs, r)
        push!(ptrs, Base.unsafe_convert(Ptr{Cvoid}, Base.pointer_from_objref(r)))
    end
    gx, gy, gz = blocks
    stream = AMDGPU.stream()
    GC.@preserve refs ptrs begin
        _hip_check(AMDGPU.HIP.hipModuleLaunchKernel(
            fun, gx, gy, gz, threads, 1, 1, shmem, stream.stream,
            pointer(ptrs), C_NULL))
    end
    return nothing
end

end # module
