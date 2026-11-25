module TritonShim

using CUDA
using cuTile
import cuTile: AbstractKernel
import cuTile as ct
import GPUCompiler
using ..TritonEmitter
using ..TritonRun

mutable struct ShimKernel{F, TT} <: AbstractKernel{F, TT}
    candidates::Vector{TritonRun.TritonKernel}
    chosen::Int          # 0 = not yet tuned
    f::F
    # ABI parameter types of the flattened arguments (plus the trailing seed
    # and scratch pointers), fixed per specialization; filled on first launch
    abi_types::Union{Nothing, Vector{Any}}
    spans::Vector{Vector{TritonEmitter.SpanTerm}}  # every candidate's, checked per launch
    wide::Any            # the i64-offset ShimKernel, once a launch has needed it
    hints::NamedTuple    # explicit launch hints it was built with
    # the explicit hints and the overlapping arrays (cuTile's `alias_groups`) it
    # was compiled for, as in cuTile's `TileKernel`; `labels` is the overlap
    # pattern of those groups (`ct.overlap_labels` form), checked per launch
    opts::NamedTuple
    labels::Vector{Int}
end
function ShimKernel{F, TT}(cands, chosen, f, hints, alias_groups) where {F, TT}
    labels = ct.groups_to_labels(alias_groups, ct.array_leaves(TT.parameters))
    ShimKernel{F, TT}(cands, chosen, f, nothing, unique(reduce(vcat, (c.spans for c in cands))),
                      nothing, hints, (; hints..., alias_groups), labels)
end

# launches may run on concurrent tasks: `KERNEL_LOCK` guards `KERNEL_CACHE`
const KERNEL_CACHE = Dict{Any, Any}()
const KERNEL_LOCK = ReentrantLock()

sanitize(s) = (n = replace(s, r"[^A-Za-z0-9_]" => "_"); isempty(n) || isdigit(n[1]) ? "k_" * n : n)

function get_kernel(@nospecialize(f), @nospecialize(tt); wide_offsets::Bool=false,
                    num_ctas=nothing, occupancy=nothing, num_worker_warps=nothing,
                    alias_groups::ct.AliasGroups=(), _...)
    hints = (; num_ctas, occupancy, num_worker_warps)
    # world counter in the key: method redefinition invalidates (cuTile rides
    # CodeInstance caching for this; a Dict is enough for the harness). A
    # closure keys by value, so each set of captured values compiles anew.
    key = (f, tt, Base.get_world_counter(), wide_offsets, hints, alias_groups)
    # compiling under the lock builds each kernel once
    @lock KERNEL_LOCK get!(KERNEL_CACHE, key) do
        compile_kernel(f, tt, hints; wide_offsets, alias_groups)
    end
end

function compile_kernel(@nospecialize(f), @nospecialize(tt), hints; wide_offsets, alias_groups)
    name = sanitize(string(nameof(typeof(f))))
    # TMA off for the coverage run: legality hardening (min box bytes, stride
    # divisibility) is orthogonal to intrinsic coverage.
    cands = if haskey(ENV, "TRITON_NUM_WARPS")
        [TritonRun.triton_kernel(f, tt; name, num_warps=parse(Int, ENV["TRITON_NUM_WARPS"]),
                                 wide_offsets, hints, alias_groups)]
    else
        TritonRun.triton_kernel_candidates(f, tt; name, wide_offsets, hints, alias_groups)
    end
    return ShimKernel{typeof(f), tt}(cands, length(cands) == 1 ? 1 : 0, f, hints, alias_groups)
end

_iscuda() = TritonRun._default_target().backend == "cuda"

# `@device_code_*` reflection: cuTile reports each kernel a launch looks up to
# GPUCompiler's compile hook as a `TileJob`; report the same job for shim
# launches so the native reflection macros keep working. Without a CUDA
# device (ROCm) the job targets `REFLECTION_SM_ARCH`, the H100 the CUDA runs
# use. `JOB_SOURCES` maps a reported job back to the `(f, tt)` it was built
# from, for hooks that reflect through this backend instead (`code_triton`);
# `TileJob`s compare structurally, so it holds one entry per distinct job.
# `JOB_LOCK` guards it against launches on concurrent tasks.
const REFLECTION_SM_ARCH = v"9.0"
const JOB_SOURCES = Dict{Any, Any}()
const JOB_LOCK = ReentrantLock()
const JOB_OPTIONS = (:opt_level, :num_ctas, :occupancy, :num_worker_warps, :name)

# Whether Tile IR supports `sm_arch` at bytecode `version`: the architecture
# condition of `ct.validate_tile_ir_target` (Tile IR needs sm_80+, and each
# architecture family a minimum bytecode version), without throwing.
function tile_ir_target_supported(sm_arch, version)
    requirement = ct.tile_ir_requirement(sm_arch)
    requirement === nothing && return false
    _, min_version = requirement
    return version >= min_version
end

function report_job(@nospecialize(f), @nospecialize(tt); sm_arch=nothing,
                    alias_groups::ct.AliasGroups=(), kwargs...)
    # internal to GPUCompiler; checked so that hook-free launches skip
    # building the job
    GPUCompiler.active_compile_hook[] === nothing && return nothing
    sm_arch = something(sm_arch, _iscuda() ? ct.device_sm_arch() : REFLECTION_SM_ARCH)
    # the default bytecode version comes from probing `tileiras`; without one
    # (no CUDA toolkit), report the newest version cuTile supports
    bytecode_version = ct.tileiras_available() ? ct.bytecode_version() :
                                                 last(ct.SUPPORTED_BYTECODE_VERSIONS)
    # `tile_job` rejects targets Tile IR cannot compile for, while Triton
    # still runs the kernel there: such a launch reports no job
    tile_ir_target_supported(sm_arch, bytecode_version) || return nothing
    opts = (; (k => v for (k, v) in pairs(kwargs) if k in JOB_OPTIONS)...)
    job = ct.tile_job(f, tt; sm_arch, bytecode_version, alias_groups, opts...)
    @lock JOB_LOCK JOB_SOURCES[job] = (f, tt)
    GPUCompiler.run_compile_hook(job)
    return nothing
end

# Flatten already-converted kernel args to the Triton ABI: TileArray →
# (ptr, sizes..., strides...); ghosts skipped; primitives passed through.
function flatten_rt!(types::Vector{Any}, vals::Vector{Any}, @nospecialize(x), cuda::Bool)
    T = typeof(x)
    if x isa ct.TileArray
        ET = eltype(x)
        if cuda
            push!(types, CuPtr{ET}); push!(vals, CuPtr{ET}(UInt(x.ptr)))
        else
            push!(types, Ptr{ET}); push!(vals, Ptr{ET}(x.ptr))
        end
        # sizes/strides carry the TileArray's index type (Int32 or Int64)
        for s in x.sizes;   push!(types, typeof(s)); push!(vals, s); end
        for s in x.strides; push!(types, typeof(s)); push!(vals, s); end
    elseif Base.issingletontype(T) || T <: Type   # ghosts and type objects (compile-time constants)
        # ghost (Constant etc.) — contributes nothing
    elseif isprimitivetype(T)
        push!(types, T); push!(vals, x)
    else
        for i in 1:fieldcount(T)
            flatten_rt!(types, vals, getfield(x, i), cuda)
        end
    end
    return
end

function _launch_one(inner::TritonRun.TritonKernel, types, vals, g)
    scratch, ptr = TritonRun.global_scratch(inner, g)
    if scratch !== nothing
        vals = copy(vals)
        vals[end - 1] = ptr
    end
    GC.@preserve scratch begin
        launchf = TritonRun._raw_launch[] === nothing ? TritonRun._cuda_launch :
                                                        TritonRun._raw_launch[]
        launchf(inner.fun, types, vals;
                threads=inner.num_warps * inner.warp_size, blocks=g, shmem=inner.shared)
    end
    return nothing
end

_device_sync() =
    (TritonRun._sync[] === nothing ? CUDA.synchronize : TritonRun._sync[])()

# Racing candidates reruns the kernel, and a kernel may read-modify-write its
# arguments (accumulation, spin locks, atomics). Snapshot every TileArray
# argument's memory extent so each candidate — and the final real launch —
# sees pristine inputs; the net effect is then exactly one application.
const SNAPSHOT_LIMIT = 2 << 30  # fall back to no racing beyond 2 GiB

function _arg_regions(args)
    regions = Tuple{UInt, Int}[]
    for a in args
        _arg_regions!(regions, a)
    end
    return regions
end

function _arg_regions!(regions, @nospecialize(x))
    T = typeof(x)
    if x isa ct.TileArray
        # strided extent in elements (holes included; restoring them is
        # harmless); a negative stride extends the region below the pointer
        lo = hi = 0
        for (sz, st) in zip(x.sizes, x.strides)
            sz == 0 && return regions
            span = (Int(sz) - 1) * Int(st)
            span < 0 ? (lo += span) : (hi += span)
        end
        esz = sizeof(eltype(x))
        push!(regions, (UInt(x.ptr) + lo * esz, (hi - lo + 1) * esz))
    elseif !Base.issingletontype(T) && !isprimitivetype(T)
        for i in 1:fieldcount(T)
            _arg_regions!(regions, getfield(x, i))
        end
    end
    return regions
end

function _snapshot(regions)
    snaps = Vector{Any}(undef, length(regions))
    for (i, (addr, n)) in enumerate(regions)
        owner, buf = TritonRun.dev_alloc(n)
        TritonRun.dev_copy(buf, addr, n)
        snaps[i] = (owner, buf)
    end
    return snaps
end

function _restore!(regions, snaps)
    for (i, (addr, n)) in enumerate(regions)
        TritonRun.dev_copy(addr, snaps[i][2], n)
    end
    return nothing
end

# Leaf values of one kernel argument in ABI order, as a tuple: the traversal
# is unrolled per argument type so a launch does no dynamic field walking.
@generated function flatten_leaves(x, ::Val{cuda}) where {cuda}
    T = x
    if T <: ct.TileArray
        ET = eltype(T)
        ptr = cuda ? :(CuPtr{$ET}(UInt(x.ptr))) : :(Ptr{$ET}(x.ptr))
        return :((($ptr,)..., x.sizes..., x.strides...))
    elseif Base.issingletontype(T) || T <: Type   # ghosts and type objects (compile-time constants)
        return :(())
    elseif isprimitivetype(T)
        return :((x,))
    else
        parts = [:(flatten_leaves(getfield(x, $i), Val(cuda))) for i in 1:fieldcount(T)]
        return :(tuple($([:($p...) for p in parts]...)))
    end
end

function (k::ShimKernel{F, TT})(args...; blocks=1, threads=1, convert=Val(false),
                                kwargs...) where {F, TT}
    cuda = _iscuda()
    PT = cuda ? CuPtr{Cvoid} : Ptr{Cvoid}
    NULLP = cuda ? CU_NULL : Ptr{Cvoid}(0)
    if k.abi_types === nothing
        types = Any[]; vals = Any[]
        # a closure's captured values come first, as in cuTile's ABI
        flatten_rt!(types, vals, k.f, cuda)
        for a in args
            flatten_rt!(types, vals, a, cuda)
        end
        push!(types, UInt32)   # KernelState seed
        push!(types, PT)       # global scratch
        push!(types, PT)       # profile scratch
        k.abi_types = types
    end
    # arrays that overlap where the kernel assumed otherwise (or the reverse)
    # need the variant compiled for their overlap pattern, as in cuTile
    labels = ct.overlap_labels(args...)
    if !ct.same_labels(k.labels, labels)
        groups = ct.labels_to_groups(labels, Tuple(ct.array_leaves(TT.parameters)))
        variant = get_kernel(k.f, TT; k.hints..., alias_groups=groups)
        return variant(args...; blocks, threads, convert, kwargs...)
    end
    types = k.abi_types
    leaves = flatten_leaves((k.f, args...), Val(cuda))
    vals = Any[leaves..., Base.rand(UInt32), NULLP, NULLP]
    if !TritonRun.spans_fit(k.spans, vals)
        # some view's in-tile offsets would overflow i32 for these arguments
        k.wide === nothing && (k.wide = get_kernel(k.f, TT; wide_offsets=true, k.opts...))
        return k.wide(args...; blocks, threads, convert, kwargs...)
    end
    length(vals) == length(types) ||
        error("kernel argument flattening produced $(length(vals)) values for $(length(types)) ABI slots")
    g = blocks isa Integer ? (Int(blocks), 1, 1) :
        length(blocks) == 2 ? (blocks[1], blocks[2], 1) : Tuple(blocks)
    if k.chosen == 0
        # first launch: race the candidates and keep the winner for this
        # specialization. Argument memory is snapshotted and restored around
        # every launch so rerunning is safe even for kernels that
        # read-modify-write their arguments (accumulation, locks, atomics).
        regions = _arg_regions((k.f, args...))
        total = sum(last, regions; init=0)
        if total > SNAPSHOT_LIMIT
            # cannot restore: don't rerun, fall back to the first candidate
            # (candidate order puts the heuristic default first)
            k.chosen = 1
        else
            snaps = _snapshot(regions)
            best = 1; best_t = Inf
            for (i, cand) in enumerate(k.candidates)
                _launch_one(cand, types, vals, g)  # warmup/compile caches
                _device_sync()
                t0 = time_ns()
                _launch_one(cand, types, vals, g)
                _device_sync()
                t = time_ns() - t0
                t < best_t && (best_t = t; best = i)
                _restore!(regions, snaps)
            end
            _device_sync()
            k.chosen = best
        end
    end
    _launch_one(k.candidates[k.chosen], types, vals, g)
    return nothing
end

end # module TritonShim



"""
    install_shim!()

Route ALL cuTile compilation through the Triton backend by overwriting
`cuTile.cufunction` and `cuTile.launch` (deliberate, opt-in piracy — the
whole point of the shim).
"""
function install_shim!()
    @eval TritonShim.ct function cufunction(@nospecialize(f), tt::Type{<:Tuple}=Tuple{}; kwargs...)
        # @device_code_* reflection rides GPUCompiler's compile hook; keep it
        # working (the hook runs the native pipeline, orthogonal to how we
        # compile/launch)
        $(TritonShim).report_job(f, tt; kwargs...)
        return $(TritonShim).get_kernel(f, tt; kwargs...)
    end
    # cuTile's launch defaults `stream=CUDACore.stream()`, which needs a CUDA
    # driver; shim kernels launch on the backend's current stream regardless.
    @eval TritonShim.ct function launch(@nospecialize(f), grid, args...;
                                        dependent::Bool=false, stream=nothing, kwargs...)
        f = cuTileconvert(f)
        converted = map(cuTileconvert, args)
        kernel = cufunction(f, Tuple{map(Core.Typeof, converted)...}; kwargs...,
                            alias_groups=alias_groups(converted...))
        kernel(converted...; blocks=grid, dependent)
        return nothing
    end
    return nothing
end
