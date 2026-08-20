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

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_csr_kernels.jl"))

# CSC → CSR with Int32 indices (CoolPDLP's transpose trick)
function to_csr(A::SparseMatrixCSC)
    At = SparseMatrixCSC(transpose(A))
    return (rowptr=Vector{Int32}(At.colptr), colval=Vector{Int32}(At.rowval),
            nzval=copy(At.nzval))
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
    for (tm, tn, tk) in csr_tile_candidates(n)
        label = "cuTile[$(tm)×$(tn)×$(tk)]"
        spmm! = try
            build_spmm(T; tile_m=tm, tile_n=tn, tile_k=tk, beta_nz=false)
        catch err
            fail_row(case, label, err)
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
        fail_row(case, VENDOR_NAME, err)
    end
    flush(stdout)
end

function main()
    m = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1_000_000
    k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : m
    nnz_row = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
    ns = length(ARGS) >= 4 ? parse.(Int, ARGS[4:end]) : [8, 64]
    T = get(ENV, "SPMM_TYPE", "Float64") == "Float32" ? Float32 : Float64

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
