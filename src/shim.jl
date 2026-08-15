module TritonShim

using CUDA
using cuTile
import cuTile: AbstractKernel
import cuTile as ct
using ..TritonEmitter
using ..TritonRun

mutable struct ShimKernel{F, TT} <: AbstractKernel{F, TT}
    candidates::Vector{TritonRun.TritonKernel}
    chosen::Int          # 0 = not yet tuned
    f::F
end

const KERNEL_CACHE = Dict{Any, Any}()

sanitize(s) = (n = replace(s, r"[^A-Za-z0-9_]" => "_"); isempty(n) || isdigit(n[1]) ? "k_" * n : n)

function get_kernel(@nospecialize(f), @nospecialize(tt))
    # world counter in the key: method redefinition invalidates (cuTile rides
    # CodeInstance caching for this; a Dict is enough for the harness)
    key = (f, tt, Base.get_world_counter())
    haskey(KERNEL_CACHE, key) && return KERNEL_CACHE[key]
    name = sanitize(string(nameof(typeof(f))))
    # TMA off for the coverage run: legality hardening (min box bytes, stride
    # divisibility) is orthogonal to intrinsic coverage.
    cands = if haskey(ENV, "TRITON_NUM_WARPS")
        [TritonRun.triton_kernel(f, tt; name, num_warps=parse(Int, ENV["TRITON_NUM_WARPS"]))]
    else
        TritonRun.triton_kernel_candidates(f, tt; name)
    end
    k = ShimKernel{typeof(f), tt}(cands, length(cands) == 1 ? 1 : 0, f)
    KERNEL_CACHE[key] = k
    return k
end

# Flatten already-converted kernel args to the Triton ABI: TileArray →
# (ptr, sizes..., strides...); ghosts skipped; primitives passed through.
function flatten_rt!(types::Vector{Any}, vals::Vector{Any}, @nospecialize(x))
    T = typeof(x)
    if x isa ct.TileArray
        ET = eltype(x)
        push!(types, CuPtr{ET}); push!(vals, CuPtr{ET}(UInt(x.ptr)))
        for s in x.sizes;   push!(types, Int32); push!(vals, s); end
        for s in x.strides; push!(types, Int32); push!(vals, s); end
    elseif Base.issingletontype(T)
        # ghost (Constant etc.) — contributes nothing
    elseif isprimitivetype(T)
        push!(types, T); push!(vals, x)
    else
        for i in 1:fieldcount(T)
            flatten_rt!(types, vals, getfield(x, i))
        end
    end
    return
end

function _launch_one(inner::TritonRun.TritonKernel, types, vals, g)
    scratch = nothing
    if inner.global_scratch_size > 0
        scratch = CuArray{UInt8}(undef, prod(g) * inner.global_scratch_size)
        vals = copy(vals)
        vals[end - 1] = reinterpret(CuPtr{Cvoid}, pointer(scratch))
    end
    GC.@preserve scratch begin
        cudacall(inner.fun, Tuple{types...}, vals...;
                 threads=inner.num_warps * inner.warp_size, blocks=g, shmem=inner.shared)
    end
    return nothing
end

function (k::ShimKernel)(args...; blocks=1, threads=1, convert=Val(false), kwargs...)
    types = Any[]; vals = Any[]
    for a in args
        flatten_rt!(types, vals, a)
    end
    push!(types, UInt32); push!(vals, Base.rand(UInt32))  # KernelState seed
    push!(types, CuPtr{Cvoid}); push!(vals, CU_NULL)   # global scratch
    push!(types, CuPtr{Cvoid}); push!(vals, CU_NULL)   # profile scratch
    g = blocks isa Integer ? (Int(blocks), 1, 1) :
        length(blocks) == 2 ? (blocks[1], blocks[2], 1) : Tuple(blocks)
    if k.chosen == 0
        # first launch: race the candidates (dot kernels without atomics are
        # rerun-safe), keep the winner for this specialization
        best = 1; best_t = Inf
        for (i, cand) in enumerate(k.candidates)
            _launch_one(cand, types, vals, g)  # warmup/compile caches
            t = CUDA.@elapsed _launch_one(cand, types, vals, g)
            t < best_t && (best_t = t; best = i)
        end
        k.chosen = best
    end
    _launch_one(k.candidates[k.chosen], types, vals, g)
    return nothing
end

end # module TritonShim



"""
    install_shim!()

Route ALL cuTile compilation through the Triton backend by overwriting
`cuTile.cufunction` (deliberate, opt-in piracy — the whole point of the shim).
"""
function install_shim!()
    @eval TritonShim.ct function cufunction(@nospecialize(f), tt::Type{<:Tuple}=Tuple{}; kwargs...)
        return $(TritonShim).get_kernel(f, tt)
    end
    return nothing
end
