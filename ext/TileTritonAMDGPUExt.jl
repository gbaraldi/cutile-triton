# Validated on MI300A (gfx942, ROCm 6.4.2): full standalone suite passes.
# The HIP launch ABI mirrors the CUDA driver one (array of pointers to the
# argument values; triton's hip metadata provides warp_size/shared).
module TileTritonAMDGPUExt

using AMDGPU
using TileTriton
using TileTriton.TritonRun: TritonRun, set_target!, _load_module, _raw_launch, _sync,
                            _dev_alloc, _dev_copy
import cuTile
import Random

# cuTile's generic TileArray(arr) refuses plain-`Ptr` arrays as a host-memory
# guard; ROCArray hands out plain `Ptr` for device memory (HIP unified
# addressing), so identify it, and its views and wrappers, explicitly. This is
# what lets unmodified cuTile launches (`cuTile.launch`, the test suite) run
# on ROCm through the shim. `use_rocm!` extends the guard to other arrays.
cuTile.device_pointer(arr::AMDGPU.AnyROCArray{T}) where {T} = Ptr{T}(pointer(arr))

function _is_device_memory(ptr::Ptr)
    try
        AMDGPU.Mem.attributes(ptr).type == AMDGPU.HIP.hipMemoryTypeDevice
    catch
        false   # not memory HIP knows about
    end
end

# cuTile's host RNG API is written against CuArray; the same fills for
# ROCArray (mirroring cuTile's methods, counter accounting included).
for (f!, kernel, Ts) in ((:rand!, :rand_fill_kernel, :RandTypes),
                         (:randn!, :randn_fill_kernel, :RandnTypes),
                         (:randexp!, :randexp_fill_kernel, :RandnTypes))
    @eval function Random.$f!(rng::cuTile.RNG, A::ROCArray{T}) where {T<:cuTile.$Ts}
        n = length(A)
        n == 0 && return A
        n_blocks = cld(n, cuTile.RAND_FILL_TILE)
        cuTile.launch(cuTile.$kernel, n_blocks, A, rng.seed, rng.counter)
        cuTile.advance_counter!(rng, UInt32(n_blocks * cuTile.RAND_FILL_TILE))
        return A
    end
    @eval cuTile.$f!(A::ROCArray{<:cuTile.$Ts}) = Random.$f!(cuTile.get_global_rng(), A)
end

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
    _dev_alloc[] = _hip_alloc
    _dev_copy[] = _hip_copy
    # triton 3.7.1's buffer-atomics lowering emits the nonexistent LLVM
    # intrinsic ...raw.ptr.buffer.atomic.exch (the real one is *.swap), which
    # kills every atomic_xchg at compile. Buffer loads/stores stay on.
    TritonRun._py_setenv!("AMDGCN_USE_BUFFER_ATOMICS", "0")
    _retarget_cutile!()
    @info "TileTriton targeting ROCm" arch wavefront=ws
    return nothing
end

# Overwrites of cuTile methods that assume CUDA memory (runtime, like the
# shim's: method overwriting is not allowed while precompiling).
function _retarget_cutile!()
    # any array whose pointer is ROCm device memory, e.g. a custom wrapper
    @eval cuTile function device_pointer(arr::AbstractArray{T}) where {T}
        ptr = pointer(arr)
        ptr isa Ptr && !$_is_device_memory(ptr) &&
            throw(ArgumentError("cannot create a TileArray from host memory (" *
                                string(typeof(arr)) * ")"))
        return reinterpret(Ptr{T}, ptr)
    end
    # the allocating host RNG calls build ROCArrays
    for (f, f!, Ts) in ((:rand, :rand!, :RandTypes), (:randn, :randn!, :RandnTypes),
                        (:randexp, :randexp!, :RandnTypes))
        @eval cuTile Random.$f(rng::RNG, ::Type{T}, dims::Dims) where {T<:$Ts} =
            Random.$f!(rng, $ROCArray{T}(undef, dims))
    end
    return nothing
end

function _hip_check(status)
    status == AMDGPU.HIP.hipSuccess ||
        error("HIP error: $(unsafe_string(AMDGPU.HIP.hipGetErrorString(status)))")
end

# device scratch and device-to-device copies on the launch stream, for the
# shim's argument snapshots
function _hip_alloc(nbytes)
    buf = ROCArray{UInt8}(undef, nbytes)
    return buf, UInt(pointer(buf))
end
_hip_copy(dst, src, nbytes) = _hip_check(AMDGPU.HIP.hipMemcpyDtoDAsync(
    Ptr{Cvoid}(dst), Ptr{Cvoid}(src), nbytes, AMDGPU.stream().stream))

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
