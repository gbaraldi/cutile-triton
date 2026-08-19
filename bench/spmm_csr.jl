# CSR SpMM (C = α·A·B + β·C) with cuTile tile kernels through TileTriton,
# benchmarked against CoolPDLP's KernelAbstractions row-per-thread kernel and
# the vendor sparse library (cuSPARSE / rocSPARSE). Runs on NVIDIA and AMD.
#
#   julia --project=. bench/spmm_csr.jl [m] [k] [nnz_per_row] [n...]
#
# Defaults: m = k = 1_000_000, ~10 nonzeros/row, n ∈ {8, 64}.
# Environment knobs:
#   SPMM_BACKEND=cuda|rocm   force the backend (default: cuda if functional,
#                            else rocm — needs AMDGPU in the environment:
#                            julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")')
#   SPMM_TYPE=Float32        element type (default Float64)
#
# Prints RESULT rows like bench/examples.jl:
#   RESULT <case> <impl> <min µs> <GFLOP/s>

using LinearAlgebra, SparseArrays, Random, Printf

const BACKEND = let req = lowercase(get(ENV, "SPMM_BACKEND", "auto"))
    if req == "auto"
        ok = try
            @eval using CUDA
            CUDA.functional()
        catch
            false
        end
        ok ? "cuda" : "rocm"
    elseif req in ("cuda", "rocm")
        req
    else
        error("SPMM_BACKEND must be cuda or rocm")
    end
end

if BACKEND == "cuda"
    using CUDA
    const GPUArr = CuArray
    device_sync() = CUDA.synchronize()
    device_name() = CUDA.name(CUDA.device())
else
    using AMDGPU
    const GPUArr = ROCArray
    device_sync() = AMDGPU.synchronize()
    device_name() = AMDGPU.HIP.name(AMDGPU.device())
end

using TileTriton
using TileTriton.TritonRun
import cuTile as ct
BACKEND == "rocm" && TileTriton.use_rocm!()

include(joinpath(@__DIR__, "spmm_csr_kernels.jl"))

# CoolPDLP baseline (src/utils/mat_csr.jl spmm_csr!): one thread per (row,
# rhs column). KernelAbstractions is an indirect dependency here, so load it
# by UUID rather than adding it to the project.
const KA = Base.require(Base.PkgId(
    Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"), "KernelAbstractions"))
using .KA  # for @kernel/@index

@kernel function spmm_csr_ka!(c, A_rowptr, A_colval, A_nzval, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    s = zero(eltype(c))
    for k in A_rowptr[i]:(A_rowptr[i + Int32(1)] - Int32(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j, batch_idx]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function vendor_csr(rowptr, colval, nzval, m, k)
    if BACKEND == "cuda"
        SP = Base.require(Base.PkgId(
            Base.UUID("b26da814-b3bc-49ef-b0ee-c816305aa060"), "cuSPARSE"))
        return SP.CuSparseMatrixCSR(rowptr, colval, nzval, (m, k))
    else
        return AMDGPU.rocSPARSE.ROCSparseMatrixCSR(rowptr, colval, nzval, (m, k))
    end
end
const VENDOR_NAME = BACKEND == "cuda" ? "cuSPARSE" : "rocSPARSE"

# ---------------------------------------------------------------------------

function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup
        f()
    end
    device_sync()
    best = Inf
    for _ in 1:nruns
        device_sync()
        t0 = time_ns()
        f()
        device_sync()
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

# CSC → CSR with Int32 indices (CoolPDLP's transpose trick)
function to_csr(A::SparseMatrixCSC)
    At = SparseMatrixCSC(transpose(A))
    return (rowptr=Vector{Int32}(At.colptr), colval=Vector{Int32}(At.rowval),
            nzval=copy(At.nzval))
end

function check(name, Cd, Cref; rtol)
    ok = isapprox(Array(Cd), Cref; rtol)
    ok || println("CHECK FAILED\t$name (max abs err ",
                  maximum(abs.(Array(Cd) .- Cref)), ")")
    return ok
end

result_row(case, impl, t, flops) =
    println("RESULT\t$case\t$impl\t$(round(t * 1e6; digits=1)) µs\t",
            round(flops / t / 1e9; digits=1), " GFLOP/s")

# Candidate (tile_m, tile_n, tile_k) configs; tile_m == 1 is the
# row-per-program kernel. Register footprint capped at 4096 gathered
# B elements per program.
function tile_candidates(n)
    tn = clamp(nextpow(2, n), 4, 64)
    cands = [(1, tn, 32), (1, tn, 64)]
    for tm in (8, 32), tk in (16, 32)
        tm * tn * tk <= 4096 && push!(cands, (tm, tn, tk))
    end
    return cands
end

function bench_case(::Type{T}, A, dcsr, n, flops; rng) where {T}
    m, k = size(A)
    case = "spmm m=$m k=$k nnz=$(nnz(A)) n=$n $(T)"
    rtol = sqrt(eps(T)) * 100

    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    Cref = A * Bh                              # α=1, β=0
    α2, β2 = T(2.5), T(-0.5)
    Cref2 = α2 .* Cref .+ β2 .* C0h            # general α, β
    Bd = GPUArr(Bh)
    Cd = GPUArr{T}(undef, m, n)
    (; rowptr, colval, nzval) = dcsr
    args = (rowptr, colval, nzval, Bd)

    # --- cuTile/TileTriton: tune over tile candidates, then verify + time ---
    best = nothing
    for (tm, tn, tk) in tile_candidates(n)
        label = "cuTile[$(tm)×$(tn)×$(tk)]"
        spmm! = try
            build_spmm(T; tile_m=tm, tile_n=tn, tile_k=tk, beta_nz=false)
        catch err
            println("RESULT\t$case\t$label\tFAILED\t",
                    replace(first(sprint(showerror, err), 100), "\n" => " "))
            continue
        end
        fill!(Cd, T(NaN))
        spmm!(Cd, args..., 1, 0)
        device_sync()
        check(label, Cd, Cref; rtol) || continue
        t = timeit(() -> spmm!(Cd, args..., 1, 0); warmup=2, nruns=5)
        result_row(case, label, t, flops)
        (best === nothing || t < best[1]) && (best = (t, tm, tn, tk, spmm!))
    end
    if best !== nothing
        (_, tm, tn, tk, spmm_best!) = best
        t = timeit(() -> spmm_best!(Cd, args..., 1, 0))
        result_row(case, "cuTile[$(tm)×$(tn)×$(tk)] best", t, flops)
        # the general α/β path (reads C) with the winning tiles
        spmm_ab! = build_spmm(T; tile_m=tm, tile_n=tn, tile_k=tk, beta_nz=true)
        copyto!(Cd, C0h)
        spmm_ab!(Cd, args..., α2, β2)
        device_sync()
        check("cuTile[$(tm)×$(tn)×$(tk)] αβ", Cd, Cref2; rtol)
        t = timeit(() -> spmm_ab!(Cd, args..., α2, β2))
        result_row(case, "cuTile[$(tm)×$(tn)×$(tk)] best αβ", t, flops)
    end

    # --- CoolPDLP KernelAbstractions baseline ---
    ka_kernel! = spmm_csr_ka!(KA.get_backend(Cd))
    ka!(α, β) = ka_kernel!(Cd, rowptr, colval, nzval, Bd, T(α), T(β); ndrange=(m, n))
    fill!(Cd, 0)                     # β=0 still reads C: keep it finite
    ka!(1, 0)
    device_sync()
    if check("KernelAbstractions", Cd, Cref; rtol)
        t = timeit(() -> ka!(1, 0))
        result_row(case, "KernelAbstractions", t, flops)
        copyto!(Cd, C0h)
        ka!(α2, β2)
        device_sync()
        check("KernelAbstractions αβ", Cd, Cref2; rtol)
    end

    # --- vendor sparse library ---
    try
        Av = vendor_csr(rowptr, colval, nzval, m, k)
        fill!(Cd, 0)
        mul!(Cd, Av, Bd, one(T), zero(T))
        device_sync()
        if check(VENDOR_NAME, Cd, Cref; rtol)
            t = timeit(() -> mul!(Cd, Av, Bd, one(T), zero(T)))
            result_row(case, VENDOR_NAME, t, flops)
        end
    catch err
        println("RESULT\t$case\t$VENDOR_NAME\tFAILED\t",
                replace(first(sprint(showerror, err), 100), "\n" => " "))
    end
    flush(stdout)
end

function main()
    m = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
    k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : m
    nnz_row = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
    ns = length(ARGS) >= 4 ? parse.(Int, ARGS[4:end]) : [8, 64]
    T = Symbol(get(ENV, "SPMM_TYPE", "Float64")) === :Float32 ? Float32 : Float64

    println("# spmm_csr bench: $BACKEND ($(device_name())), $T, ",
            "A = $m×$k with ~$nnz_row nnz/row, n = $ns")
    rng = MersenneTwister(42)
    A = sprand(rng, T, m, k, nnz_row / k)
    csr = to_csr(A)
    dcsr = (rowptr=GPUArr(csr.rowptr), colval=GPUArr(csr.colval),
            nzval=GPUArr(csr.nzval))
    for n in ns
        bench_case(T, A, dcsr, n, 2.0 * nnz(A) * n; rng)
    end
    println("SPMM BENCH DONE ($BACKEND)")
end

main()
