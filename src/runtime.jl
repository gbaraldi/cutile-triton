# TritonRun: compile emitted TTIR via the CondaPkg-pinned triton (in-process
# through PythonCall) and launch with CUDA.jl. Self-contained: Project.toml +
# CondaPkg.toml fully specify both the Julia and Python sides.
module TritonRun

using CUDA
using PythonCall
using ..TritonEmitter: emit_ttir, ArgSpec
import cuTile as ct

export triton_kernel, launch!, code_triton, set_target!

"Compilation/launch target. `backend` is \"cuda\" or \"hip\"; `arch` is the
sm number (Int) or gfx arch (String); `binkey` the asm-dict key."
struct TargetInfo
    backend::String
    arch::Union{Int,String}
    warp_size::Int
    binkey::String
end

const ACTIVE_TARGET = Ref{Union{Nothing,TargetInfo}}(nothing)
set_target!(t::TargetInfo) = (ACTIVE_TARGET[] = t)
set_target!(backend, arch, warp_size) =
    set_target!(TargetInfo(backend, arch, warp_size, backend == "cuda" ? "cubin" : "hsaco"))

function _default_target()
    t = ACTIVE_TARGET[]
    t !== nothing && return t
    # queried once: every launch consults the target, and the device
    # attribute queries cost microseconds
    sm = CUDA.capability(CUDA.device())
    return set_target!(TargetInfo("cuda", sm.major * 10 + sm.minor, 32, "cubin"))
end

# pluggable module-load and raw-launch (the AMDGPU extension replaces these
# when a hip target is active)
const _load_module = Ref{Any}(nothing)   # (bin::Vector{UInt8}, name, shared) -> (mod, fun)
const _raw_launch = Ref{Any}(nothing)    # (fun, types, vals; threads, blocks, shmem)
const _sync = Ref{Any}(nothing)          # () -> device synchronize
const _dev_alloc = Ref{Any}(nothing)     # (nbytes) -> (owner, addr::UInt) device scratch
const _dev_copy = Ref{Any}(nothing)      # (dst::UInt, src::UInt, nbytes) -> nothing, on the launch stream

function _cuda_alloc(nbytes)
    buf = CuArray{UInt8}(undef, nbytes)
    return buf, UInt(pointer(buf))
end
_cuda_copy(dst, src, nbytes) = unsafe_copyto!(CuPtr{UInt8}(dst), CuPtr{UInt8}(src), nbytes)

dev_alloc(nbytes) = (_dev_alloc[] === nothing ? _cuda_alloc : _dev_alloc[])(nbytes)
dev_copy(dst, src, nbytes) = (_dev_copy[] === nothing ? _cuda_copy : _dev_copy[])(dst, src, nbytes)

function _cuda_load(bin, name, shared)
    mod = CuModule(bin)
    fun = CuFunction(mod, name)
    if shared > 48 * 1024
        CUDA.attributes(fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shared
    end
    return mod, fun
end
# `vals` already carry their ABI types (see `flatten_vals` and the shim's
# `flatten_leaves`), so the launch takes a concretely typed tuple: argument
# packing then specializes once per signature. `cudacall` with a runtime
# `Tuple{types...}` converts every argument dynamically, which costs ~0.5 µs
# per parameter — 15–20 µs for kernels taking many TileArrays.
function _cuda_launch(fun, types, vals; threads, blocks, shmem)
    args = Tuple(vals)
    CUDA.launch(fun, args...; threads, blocks, shmem)
end

# lazy one-time imports (the wheel import costs ~1s once per session,
# vs. per-kernel with the previous subprocess driver)
const _pycompiler = Ref{Py}()
const _pybackends = Ref{Py}()
const _triton_dir = Ref{String}()

function _triton()
    if !isassigned(_pycompiler)
        _pycompiler[] = pyimport("triton.compiler")
        _pybackends[] = pyimport("triton.backends.compiler")
        _triton_dir[] = dirname(pyconvert(String, pyimport("triton").__file__))
    end
    return _pycompiler[], _pybackends[]
end

# Python's os.environ snapshots the environment at interpreter start; triton's
# knobs read through it, so backend workarounds must set it there, not ENV.
_py_setenv!(k::String, v::String) = (pyimport("os").environ[k] = v; nothing)

"Compile a TTIR string in-process; returns the triton CompiledKernel (Py)."
function _compile_py(ttir::String, name::String, num_warps::Int, num_stages::Int)
    tc, bc = _triton()
    path = joinpath(mktempdir(), "$name.ttir")
    write(path, ttir)
    t = _default_target()
    target = bc.GPUTarget(t.backend, t.arch, t.warp_size)
    opts = pydict(Dict("num_warps" => num_warps, "num_stages" => num_stages))
    try
        return tc.compile(path; target=target, options=opts)
    catch e
        e isa PyException || rethrow()
        error("triton compile failed for $name:\n" * sprint(showerror, e))
    end
end

struct TritonKernel
    fun::Any
    mod::Any                 # keep alive
    name::String
    num_warps::Int
    warp_size::Int
    shared::Int
    global_scratch_size::Int
    argspec::Vector{ArgSpec}
    ttir::String
    abi_types::Vector{Any}   # flattened ABI parameter types, fixed per kernel
end

function jint(md, key, default)
    m = match(Regex("\"$key\":\\s*(-?\\d+)"), md)
    return m === nothing ? default : parse(Int, m[1])
end

"""
    triton_kernel_candidates(f, argtypes; name, use_tma=true) -> Vector{TritonKernel}

A 4- and an 8-warp build for first-launch autotuning (mirrors
`@triton.autotune`: measure once per specialization, remember the winner);
tensor-core kernels additionally race load style × hints. The shim's racer
snapshots and restores argument memory, so rerunning is safe for any
kernel, atomics and read-modify-write included; when it cannot snapshot it
falls back to the first candidate, so each list leads with the heuristic
default (4 warps for memory-bound kernels, 8 for tensor-core ones —
8 warps wins on starved grids and loop-heavy reductions, e.g. sm_120
layernorm bwd_dwdb 101→59µs and bwd_dx 414→284µs).
"""
# Triton's pre-Hopper lowering of TMA descriptor loads/stores fails for
# 16-bit floats (`ConvertTritonGPUToLLVM` → "PassManager::run failed");
# plain pointer accesses compile fine, so before sm_90 descriptors are
# skipped whenever a 16-bit float array reaches the kernel (recursively:
# TileArrays arrive inside structs such as SplitRangeCSRMatrix). Hopper and
# later lower them fine, and there they are the fast path.
function has_16bit_float_array(@nospecialize(T), seen = Base.IdSet{Any}())
    T in seen && return false
    push!(seen, T)
    T isa DataType || return false
    if T <: ct.TileArray
        ET = eltype(T)
        return ET <: AbstractFloat && sizeof(ET) == 2
    end
    return any(FT -> has_16bit_float_array(FT, seen), fieldtypes(T))
end
function tma_eligible(@nospecialize(argtypes), target::TargetInfo=_default_target())
    target.backend == "cuda" && target.arch isa Int && target.arch >= 90 && return true
    return !any(has_16bit_float_array, argtypes isa Type ? argtypes.parameters : argtypes)
end

function triton_kernel_candidates(@nospecialize(f), @nospecialize(argtypes);
                                  name::String, use_tma::Bool=true)
    use_tma &= _default_target().backend == "cuda"  # descriptors are NVIDIA-only
    use_tma &= tma_eligible(argtypes)
    ttir, argspec, meta = emit_ttir(f, argtypes; name, use_tma)
    if !meta.has_dot
        return [compile_kernel(ttir, argspec; name, num_warps=w) for w in (4, 8)]
    end
    # dot kernels: TMA-vs-pointer and argument hints both interact with the
    # tensor-core pipeline in shape-dependent ways — race load style × hints
    # × warps (unique TTIRs only; e.g. TMA-ineligible kernels dedupe)
    variants = unique([ttir,
                       emit_ttir(f, argtypes; name, use_tma=false, hints=true)[1],
                       emit_ttir(f, argtypes; name, use_tma=false, hints=false)[1]])
    return [compile_kernel(t, argspec; name, num_warps=w)
            for t in variants for w in (8, 4)]
end

function triton_kernel(@nospecialize(f), @nospecialize(argtypes);
                       name::String, num_warps::Union{Int,Nothing}=nothing,
                       num_stages::Union{Int,Nothing}=nothing, use_tma::Bool=true)
    use_tma &= _default_target().backend == "cuda"  # descriptors are NVIDIA-only
    use_tma &= tma_eligible(argtypes)
    ttir, argspec, meta = emit_ttir(f, argtypes; name, use_tma)
    # schedule heuristic mirroring tileiras' observed choices: tensor-core
    # kernels get 8 warps, memory-bound kernels 4
    num_warps = num_warps !== nothing ? num_warps : (meta.has_dot ? 8 : 4)
    return compile_kernel(ttir, argspec; name, num_warps, num_stages)
end

# triton's own backend defaults: 3 pipeline stages on NVIDIA, 2 on AMD (64KB LDS)
_default_stages() = _default_target().backend == "cuda" ? 3 : 2

# Per-block dynamic shared memory limit of the current device (opt-in), or
# nothing when the backend has no queryable limit.
_shared_limit() = _default_target().backend == "cuda" ?
    CUDA.attribute(CUDA.device(),
                   CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) : nothing

function compile_kernel(ttir::String, argspec; name::String, num_warps::Int,
                        num_stages::Union{Int,Nothing}=nothing)
    explicit_stages = num_stages !== nothing
    num_stages = something(num_stages, _default_stages())
    if haskey(ENV, "TRITON_DUMP_TTIR")
        mkpath(ENV["TRITON_DUMP_TTIR"])
        write(joinpath(ENV["TRITON_DUMP_TTIR"], "$(name)_w$(num_warps)_$(hash(ttir) % 10000).ttir"), ttir)
    end
    k = _compile_py(ttir, name, num_warps, num_stages)
    shared = pyconvert(Int, k.metadata.shared)
    # triton sizes its software pipeline for the tile shape, not the device:
    # a 3-stage matmul that fits an H100's 228KB can exceed a consumer GPU's
    # ~99KB opt-in limit. Drop stages until the kernel fits.
    limit = explicit_stages ? nothing : _shared_limit()
    while limit !== nothing && shared > limit && num_stages > 1
        num_stages -= 1
        k = _compile_py(ttir, name, num_warps, num_stages)
        shared = pyconvert(Int, k.metadata.shared)
    end
    t = _default_target()
    bin = pyconvert(Vector{UInt8}, k.asm[t.binkey])
    warp_size = pyconvert(Int, k.metadata.warp_size)
    # NVIDIA-only metadata field; the AMD backend has no global scratch
    # (its launcher passes NULL in that ABI slot).
    gss = pyconvert(Int, pygetattr(k.metadata, "global_scratch_size", 0))
    loadf = _load_module[] === nothing ? _cuda_load : _load_module[]
    mod, fun = loadf(bin, name, shared)
    if haskey(ENV, "TRITON_KERNEL_INFO") && fun isa CUDA.CuFunction
        # occupancy diagnostics: registers/thread and local (spill) bytes
        attrs = CUDA.attributes(fun)
        println("KERNEL\t$name\tnum_warps=$num_warps\tregs=",
                attrs[CUDA.FUNC_ATTRIBUTE_NUM_REGS], "\tlocal=",
                attrs[CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES], "\tshared=$shared")
    end
    return TritonKernel(fun, mod, name, num_warps, warp_size, shared, gss, argspec, ttir,
                        abi_types(argspec))
end

# Flatten runtime arguments to the kernel ABI: TileArray → (ptr, sizes..., strides...).
"ABI parameter types for `spec`: per TileArray a pointer, its sizes and its
strides; scalars as is; then the KernelState seed and the two scratch
pointers (triton >= 3.2 ABI)."
function abi_types(spec::Vector{ArgSpec})
    cuda = _default_target().backend == "cuda"
    types = Any[]
    for s in spec
        if s.kind === :tilearray
            push!(types, cuda ? CuPtr{s.elty} : Ptr{s.elty})
            for _ in 1:(2 * s.ndims)
                push!(types, s.idxty)
            end
        else
            push!(types, s.elty)
        end
    end
    push!(types, UInt32)
    PT = cuda ? CuPtr{Cvoid} : Ptr{Cvoid}
    push!(types, PT); push!(types, PT)
    return types
end

"Argument values in the order of `abi_types(spec)`."
function flatten_vals(spec::Vector{ArgSpec}, args)
    length(spec) == length(args) || error("expected $(length(spec)) kernel arguments")
    vals = Any[]
    cuda = _default_target().backend == "cuda"
    for (s, a) in zip(spec, args)
        if s.kind === :tilearray
            a isa AbstractArray || error("expected a GPU array for tilearray argument")
            eltype(a) === s.elty || error("eltype mismatch: $(eltype(a)) vs $(s.elty)")
            ndims(a) == s.ndims || error("ndims mismatch")
            p = pointer(a)
            push!(vals, cuda ? p : Ptr{s.elty}(UInt(p)))
            for d in 1:s.ndims
                push!(vals, s.idxty(size(a, d)))
            end
            st = strides(a)
            for d in 1:s.ndims
                push!(vals, s.idxty(st[d]))
            end
        else
            push!(vals, s.elty(a))
        end
    end
    push!(vals, Base.rand(UInt32))                 # implicit KernelState seed
    NULLP = cuda ? CU_NULL : Ptr{Cvoid}(0)
    push!(vals, NULLP); push!(vals, NULLP)         # global and profile scratch
    return vals
end

"""
    launch!(k::TritonKernel, grid, args...)

`grid` is an Int or (gx, gy[, gz]); `args` are the kernel's Julia-level
arguments in order, skipping `ct.Constant` slots.
"""
function launch!(k::TritonKernel, grid, args...)
    vals = flatten_vals(k.argspec, args)
    types = k.abi_types
    g = grid isa Integer ? (Int(grid), 1, 1) : (length(grid) == 2 ? (grid[1], grid[2], 1) : grid)
    # TMA descriptors are built in global scratch memory; size it like
    # Python's launcher: grid volume × num_ctas × per-CTA scratch.
    scratch = nothing
    if k.global_scratch_size > 0
        scratch = CuArray{UInt8}(undef, prod(g) * k.global_scratch_size)
        vals[end - 1] = reinterpret(CuPtr{Cvoid}, pointer(scratch))
    end
    GC.@preserve scratch begin
        launchf = _raw_launch[] === nothing ? _cuda_launch : _raw_launch[]
        launchf(k.fun, types, vals;
                threads=k.num_warps * k.warp_size, blocks=g, shmem=k.shared)
    end
    return nothing
end

"""
    code_triton([io], f, argtypes; stage=:ttir, name="kernel",
                num_warps=4, num_stages=backend default, use_tma=true)

Reflection for the Triton backend, mirroring cuTile's `@device_code_*` family.
`stage` is one of `:ttir` (as emitted), `:ttgir`, `:llir`, `:ptx`, `:sass`.
"""
function code_triton(io::IO, @nospecialize(f), @nospecialize(argtypes);
                     stage::Symbol=:ttir, name::String="kernel",
                     num_warps::Int=4, num_stages::Union{Int,Nothing}=nothing, use_tma::Bool=true)
    use_tma &= _default_target().backend == "cuda"  # descriptors are NVIDIA-only
    ttir, _ = emit_ttir(f, argtypes; name, use_tma)
    if stage === :ttir
        print(io, ttir)
        return nothing
    end
    k = _compile_py(ttir, name, num_warps, something(num_stages, _default_stages()))
    if stage === :sass
        cubin_path = joinpath(mktempdir(), "$name.cubin")
        write(cubin_path, pyconvert(Vector{UInt8}, k.asm["cubin"]))
        nvdisasm = joinpath(_triton_dir[], "backends", "nvidia", "bin", "nvdisasm")
        print(io, read(`$nvdisasm -c $cubin_path`, String))
    elseif stage in (:ttgir, :llir, :ptx)
        print(io, pyconvert(String, k.asm[String(stage)]))
    else
        error("code_triton: unknown stage :$stage (use :ttir/:ttgir/:llir/:ptx/:sass)")
    end
    return nothing
end
code_triton(@nospecialize(f), @nospecialize(argtypes); kwargs...) =
    code_triton(stdout, f, argtypes; kwargs...)

end # module
