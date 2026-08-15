# TritonRun: compile emitted TTIR via the CondaPkg-pinned triton (in-process
# through PythonCall) and launch with CUDA.jl. Self-contained: Project.toml +
# CondaPkg.toml fully specify both the Julia and Python sides.
module TritonRun

using CUDA
using PythonCall
using ..TritonEmitter: emit_ttir, ArgSpec

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
    sm = CUDA.capability(CUDA.device())
    TargetInfo("cuda", sm.major * 10 + sm.minor, 32, "cubin")
end

# pluggable module-load and raw-launch (the AMDGPU extension replaces these
# when a hip target is active)
const _load_module = Ref{Any}(nothing)   # (bin::Vector{UInt8}, name, shared) -> (mod, fun)
const _raw_launch = Ref{Any}(nothing)    # (fun, types, vals; threads, blocks, shmem)

function _cuda_load(bin, name, shared)
    mod = CuModule(bin)
    fun = CuFunction(mod, name)
    if shared > 48 * 1024
        CUDA.attributes(fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shared
    end
    return mod, fun
end
_cuda_launch(fun, types, vals; threads, blocks, shmem) =
    cudacall(fun, Tuple{types...}, vals...; threads, blocks, shmem)

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
end

function jint(md, key, default)
    m = match(Regex("\"$key\":\\s*(-?\\d+)"), md)
    return m === nothing ? default : parse(Int, m[1])
end

"""
    triton_kernel_candidates(f, argtypes; name, use_tma=true) -> Vector{TritonKernel}

One kernel for memory-bound code (4 warps); for tensor-core kernels without
atomics, both a 4- and an 8-warp build for first-launch autotuning (mirrors
`@triton.autotune`: measure once per specialization, remember the winner).
"""
function triton_kernel_candidates(@nospecialize(f), @nospecialize(argtypes);
                                  name::String, use_tma::Bool=true)
    use_tma &= _default_target().backend == "cuda"  # descriptors are NVIDIA-only
    ttir, argspec, meta = emit_ttir(f, argtypes; name, use_tma)
    if !meta.has_dot
        return [compile_kernel(ttir, argspec; name, num_warps=4)]
    end
    meta.has_atomic &&
        return [compile_kernel(ttir, argspec; name, num_warps=8)]
    # dot kernels: TMA-vs-pointer and argument hints both interact with the
    # tensor-core pipeline in shape-dependent ways — race load style × hints
    # × warps (unique TTIRs only; e.g. TMA-ineligible kernels dedupe)
    variants = unique([ttir,
                       emit_ttir(f, argtypes; name, use_tma=false, hints=true)[1],
                       emit_ttir(f, argtypes; name, use_tma=false, hints=false)[1]])
    return [compile_kernel(t, argspec; name, num_warps=w)
            for t in variants for w in (4, 8)]
end

function triton_kernel(@nospecialize(f), @nospecialize(argtypes);
                       name::String, num_warps::Union{Int,Nothing}=nothing,
                       num_stages::Union{Int,Nothing}=nothing, use_tma::Bool=true)
    use_tma &= _default_target().backend == "cuda"  # descriptors are NVIDIA-only
    ttir, argspec, meta = emit_ttir(f, argtypes; name, use_tma)
    # schedule heuristic mirroring tileiras' observed choices: tensor-core
    # kernels get 8 warps, memory-bound kernels 4
    num_warps = num_warps !== nothing ? num_warps : (meta.has_dot ? 8 : 4)
    return compile_kernel(ttir, argspec; name, num_warps, num_stages)
end

# triton's own backend defaults: 3 pipeline stages on NVIDIA, 2 on AMD (64KB LDS)
_default_stages() = _default_target().backend == "cuda" ? 3 : 2

function compile_kernel(ttir::String, argspec; name::String, num_warps::Int,
                        num_stages::Union{Int,Nothing}=nothing)
    num_stages = something(num_stages, _default_stages())
    if haskey(ENV, "TRITON_DUMP_TTIR")
        mkpath(ENV["TRITON_DUMP_TTIR"])
        write(joinpath(ENV["TRITON_DUMP_TTIR"], "$(name)_w$(num_warps)_$(hash(ttir) % 10000).ttir"), ttir)
    end
    k = _compile_py(ttir, name, num_warps, num_stages)
    t = _default_target()
    bin = pyconvert(Vector{UInt8}, k.asm[t.binkey])
    shared = pyconvert(Int, k.metadata.shared)
    warp_size = pyconvert(Int, k.metadata.warp_size)
    # NVIDIA-only metadata field; the AMD backend has no global scratch
    # (its launcher passes NULL in that ABI slot).
    gss = pyconvert(Int, pygetattr(k.metadata, "global_scratch_size", 0))
    loadf = _load_module[] === nothing ? _cuda_load : _load_module[]
    mod, fun = loadf(bin, name, shared)
    return TritonKernel(fun, mod, name, num_warps, warp_size, shared, gss, argspec, ttir)
end

# Flatten runtime arguments to the kernel ABI: TileArray → (ptr, sizes..., strides...).
function flatten_args(spec::Vector{ArgSpec}, args)
    length(spec) == length(args) || error("expected $(length(spec)) kernel arguments")
    types = Any[]
    vals = Any[]
    cuda = _default_target().backend == "cuda"
    for (s, a) in zip(spec, args)
        if s.kind === :tilearray
            a isa AbstractArray || error("expected a GPU array for tilearray argument")
            eltype(a) === s.elty || error("eltype mismatch: $(eltype(a)) vs $(s.elty)")
            ndims(a) == s.ndims || error("ndims mismatch")
            p = pointer(a)
            if cuda
                push!(types, CuPtr{s.elty}); push!(vals, p)
            else
                push!(types, Ptr{s.elty}); push!(vals, Ptr{s.elty}(UInt(p)))
            end
            for d in 1:s.ndims
                push!(types, Int32); push!(vals, Int32(size(a, d)))
            end
            st = strides(a)
            for d in 1:s.ndims
                push!(types, Int32); push!(vals, Int32(st[d]))
            end
        else
            push!(types, s.elty); push!(vals, s.elty(a))
        end
    end
    # implicit KernelState seed (mirrors cuTile's ABI)
    push!(types, UInt32); push!(vals, Base.rand(UInt32))
    # trailing global-scratch and profile-scratch pointers (triton >= 3.2 ABI)
    PT = cuda ? CuPtr{Cvoid} : Ptr{Cvoid}
    NULLP = cuda ? CU_NULL : Ptr{Cvoid}(0)
    push!(types, PT); push!(vals, NULLP)
    push!(types, PT); push!(vals, NULLP)
    return Tuple{types...}, vals
end

"""
    launch!(k::TritonKernel, grid, args...)

`grid` is an Int or (gx, gy[, gz]); `args` are the kernel's Julia-level
arguments in order, skipping `ct.Constant` slots.
"""
function launch!(k::TritonKernel, grid, args...)
    tt, vals = flatten_args(k.argspec, collect(args))
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
        launchf(k.fun, collect(tt.parameters), vals;
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
    k = _compile_py(ttir, name, num_warps, num_stages)
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
