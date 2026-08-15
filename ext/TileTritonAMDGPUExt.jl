# EXPERIMENTAL — written without AMD hardware to iterate on; the compile side
# (TTIR → hsaco via the wheel's `hip` backend) is known-good from offline
# tests, the load/launch side needs validation on a ROCm box.
module TileTritonAMDGPUExt

using AMDGPU
using TileTriton
using TileTriton.TritonRun: TritonRun, set_target!, _load_module, _raw_launch

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
