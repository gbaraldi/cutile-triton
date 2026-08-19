# SpMM bench on the min-cost-flow constraint matrix (TX road network,
# node-arc incidence — the matrix from CoolPDLP2/reactant/benchmarks.jl),
# in every format we have a kernel for:
#
#   A  (nodes×arcs, ~5 nnz/row):  cuTile CSR, cuTile JDS (pm/vals),
#                                 KA jds, KA csr, cuSPARSE/rocSPARSE
#   Aᵀ (arcs×nodes, ≤2 nnz/row):  cuTile CSR, cuTile 2-per-row (pm/vals),
#                                 KA 2pr, KA csr, cuSPARSE/rocSPARSE
#
#   julia --project=. bench/spmm_flow.jl [n...]      (default n = 8, 64, 256)
#
# Needs bench/data/flow_TX{,_t,_zoo}.jls, serialized by the export script
# (SparseMatrixCSC{Float32,Int32} of A and Aᵀ plus the raw JDSMatrixPM /
# Matrix2PerRowPM arrays from MinimumCostFlows).
# Environment knobs as in bench/spmm_csr.jl: SPMM_BACKEND, SPMM_TYPE
# (default Float32 here, matching CoolPDLP).

using LinearAlgebra, SparseArrays, Random, Printf, Serialization

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
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

const KA = Base.require(Base.PkgId(
    Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"), "KernelAbstractions"))
using .KA  # for @kernel/@index

# --- KA baselines: CoolPDLP's csr kernel plus SpMM versions of the
# --- matrix_zoo kernels (one thread per (row, rhs column)) ------------------

@kernel function spmm_csr_ka!(c, A_rowptr, A_colval, A_nzval, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    s = zero(eltype(c))
    for k in A_rowptr[i]:(A_rowptr[i + Int32(1)] - Int32(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j, batch_idx]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

@kernel function spmm_jds_pm_ka!(c, colidx, iterptr, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    T = eltype(c)
    @inbounds begin
        s = zero(T)
        j = 1
        ptr = i
        while j < length(iterptr) && ptr < iterptr[j + 1]
            k = colidx[ptr]
            bₖ = b[abs(k), batch_idx]
            s += ifelse(k < 0, -bₖ, bₖ)
            ptr = i + iterptr[j += 1] - 1
        end
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

@kernel function spmm_jds_ka!(c, colidx, iterptr, vals, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    T = eltype(c)
    @inbounds begin
        s = zero(T)
        j = 1
        ptr = i
        while j < length(iterptr) && ptr < iterptr[j + 1]
            s += vals[ptr] * b[colidx[ptr], batch_idx]
            ptr = i + iterptr[j += 1] - 1
        end
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

@kernel function spmm_2pr_pm_ka!(c, colidx, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    T = eltype(c)
    @inbounds begin
        s = zero(T)
        j = colidx[1, i]
        j > 0 && (s += b[j, batch_idx])
        j = colidx[2, i]
        j > 0 && (s -= b[j, batch_idx])
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

@kernel function spmm_2pr_ka!(c, colidx, vals, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    T = eltype(c)
    @inbounds begin
        s = zero(T)
        j = colidx[1, i]
        j > 0 && (s += vals[1, i] * b[j, batch_idx])
        j = colidx[2, i]
        j > 0 && (s += vals[2, i] * b[j, batch_idx])
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
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

function check(name, Cd, Cref; rtol)
    ok = isapprox(Array(Cd), Cref; rtol)
    ok || println("CHECK FAILED\t$name (max abs err ",
                  maximum(abs.(Array(Cd) .- Cref)), ")")
    return ok
end

result_row(case, impl, t, flops) =
    println("RESULT\t$case\t$impl\t$(round(t * 1e6; digits=1)) µs\t",
            round(flops / t / 1e9; digits=1), " GFLOP/s")

"""
Verify and time one implementation. `f0(Cd)` runs the β=0 path in place,
`fab(Cd)` the general α/β path (or `nothing` to skip); `init0` is what Cd
must hold before `f0` (NaN poison for kernels that never read C, zeros for
the KA/vendor paths that do). Returns the β=0 best time (or nothing).
"""
function bench_impl(case, label, flops, Cd, ctx; f0, fab=nothing, init0=:nan)
    (; Cref, Cref2, C0h, rtol) = ctx
    T = eltype(Cd)
    init0 === :nan ? fill!(Cd, T(NaN)) : fill!(Cd, zero(T))
    f0(Cd)
    device_sync()
    check("$label", Cd, Cref; rtol) || return nothing
    t = timeit(() -> f0(Cd))
    result_row(case, label, t, flops)
    if fab !== nothing
        copyto!(Cd, C0h)
        fab(Cd)
        device_sync()
        check("$label αβ", Cd, Cref2; rtol)
        tab = timeit(() -> fab(Cd))
        result_row(case, "$label αβ", tab, flops)
    end
    flush(stdout)
    return t
end

# tile candidates, register footprint capped at 4096 gathered B elements
function csr_tile_candidates(n)
    tn = clamp(nextpow(2, n), 4, 64)
    cands = [(1, tn, 32), (1, tn, 64)]
    for tm in (8, 32), tk in (16, 32)
        tm * tn * tk <= 4096 && push!(cands, (tm, tn, tk))
    end
    return cands
end

function zoo_tile_candidates(n; per_row)
    tn = clamp(nextpow(2, n), 4, 64)
    cap = 4096 ÷ per_row              # per_row B elements gathered per row
    return [(tm, tn) for tm in (32, 64, 128, 256) if tm * tn <= cap]
end

function bench_cutile_csr(case, flops, Cd, ctx, args, n, α2, β2)
    (; T) = ctx
    best = nothing
    for (tm, tn, tk) in csr_tile_candidates(n)
        label = "cuTile csr[$(tm)×$(tn)×$(tk)]"
        spmm! = try
            build_spmm(T; tile_m=tm, tile_n=tn, tile_k=tk, beta_nz=false)
        catch err
            println("RESULT\t$case\t$label\tFAILED\t",
                    replace(first(sprint(showerror, err), 100), "\n" => " "))
            continue
        end
        t = bench_impl(case, label, flops, Cd, ctx;
                       f0=C -> spmm!(C, args..., 1, 0))
        t === nothing && continue
        (best === nothing || t < best[1]) && (best = (t, tm, tn, tk))
    end
    if best !== nothing
        (_, tm, tn, tk) = best
        spmm_ab! = build_spmm(T; tile_m=tm, tile_n=tn, tile_k=tk, beta_nz=true)
        copyto!(Cd, ctx.C0h)
        spmm_ab!(Cd, args..., α2, β2)
        device_sync()
        check("cuTile csr[$(tm)×$(tn)×$(tk)] αβ", Cd, ctx.Cref2; rtol=ctx.rtol)
        t = timeit(() -> spmm_ab!(Cd, args..., α2, β2))
        result_row(case, "cuTile csr[$(tm)×$(tn)×$(tk)] best αβ", t, flops)
    end
end

"""
Tune a zoo kernel over (tile_m, tile_n) candidates. `build(tm, tn, beta_nz)`
returns a launcher, `launch(spmm!, C, α, β)` runs it.
"""
function bench_cutile_zoo(case, name, flops, Cd, ctx, n, α2, β2;
                          build, launch, per_row)
    best = nothing
    for (tm, tn) in zoo_tile_candidates(n; per_row)
        label = "cuTile $name[$(tm)×$(tn)]"
        spmm! = try
            build(tm, tn, false)
        catch err
            println("RESULT\t$case\t$label\tFAILED\t",
                    replace(first(sprint(showerror, err), 100), "\n" => " "))
            continue
        end
        t = bench_impl(case, label, flops, Cd, ctx;
                       f0=C -> launch(spmm!, C, 1, 0))
        t === nothing && continue
        (best === nothing || t < best[1]) && (best = (t, tm, tn))
    end
    if best !== nothing
        (_, tm, tn) = best
        spmm_ab! = build(tm, tn, true)          # reads C: needs it initialized
        copyto!(Cd, ctx.C0h)
        launch(spmm_ab!, Cd, α2, β2)
        device_sync()
        check("cuTile $name[$(tm)×$(tn)] αβ", Cd, ctx.Cref2; rtol=ctx.rtol)
        t = timeit(() -> launch(spmm_ab!, Cd, α2, β2))
        result_row(case, "cuTile $name[$(tm)×$(tn)] best αβ", t, flops)
        flush(stdout)
    end
end

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = Symbol(get(ENV, "SPMM_TYPE", "Float32")) === :Float64 ? Float64 : Float32
    datadir = joinpath(@__DIR__, "data")

    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "flow_TX.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "flow_TX_t.jls")))
    zoo = deserialize(joinpath(datadir, "flow_TX_zoo.jls"))
    m, k = size(A)
    println("# spmm_flow bench: $BACKEND ($(device_name())), $T, ",
            "A = $m×$k with nnz=$(nnz(A)) (TX road incidence), n = $ns")
    rng = MersenneTwister(42)
    α2, β2 = T(2.5), T(-0.5)
    rtol = sqrt(eps(T)) * 100

    # CSR of A is the CSC of Aᵀ and vice versa
    csrA = (rowptr=GPUArr(At.colptr), colval=GPUArr(At.rowval),
            nzval=GPUArr(T.(At.nzval)))
    csrAt = (rowptr=GPUArr(A.colptr), colval=GPUArr(A.rowval),
             nzval=GPUArr(T.(A.nzval)))

    # JDS arrays of A (identity row map); cuTile needs a trailing sentinel
    jds_col = GPUArr(zoo.jds_colidx)
    jds_col_abs = GPUArr(abs.(zoo.jds_colidx))
    jds_nz = GPUArr(T.(ifelse.(zoo.jds_colidx .< 0, -1, 1)))
    jds_iter = GPUArr(zoo.jds_iterptr)
    jds_iter_pad = GPUArr(push!(copy(zoo.jds_iterptr), zoo.jds_iterptr[end]))
    @assert zoo.jds_nrows == m && zoo.jds_ncols == k

    # 2-per-row arrays of Aᵀ (flat for cuTile, 2×m for KA)
    tpr_col2 = GPUArr(zoo.tpr_colidx)
    tpr_col = vec(tpr_col2)
    tpr_vals2 = GPUArr(repeat(T[1, -1], 1, size(zoo.tpr_colidx, 2)))
    tpr_vals = vec(tpr_vals2)
    @assert size(zoo.tpr_colidx, 2) == k && zoo.tpr_ncols == m

    for (mat, csr, Amat) in (("A", csrA, A), ("At", csrAt, At))
        mm, kk = size(Amat)
        for n in ns
            case = "flow $mat n=$n $T"
            flops = 2.0 * nnz(Amat) * n
            Bh = rand(rng, T, kk, n)
            C0h = rand(rng, T, mm, n)
            Cref = Amat * Bh
            Cref2 = α2 .* Cref .+ β2 .* C0h
            Bd = GPUArr(Bh)
            Cd = GPUArr{T}(undef, mm, n)
            ctx = (; T, Cref, Cref2, C0h, rtol)

            # cuTile CSR
            bench_cutile_csr(case, flops, Cd, ctx,
                             (csr.rowptr, csr.colval, csr.nzval, Bd), n, α2, β2)

            if mat == "A"
                # cuTile JDS
                bench_cutile_zoo(case, "jds pm", flops, Cd, ctx, n, α2, β2;
                    per_row=1,
                    build=(tm, tn, bnz) -> build_spmm_jds(T; tile_m=tm, tile_n=tn,
                                                          pm=true, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, jds_col, jds_iter_pad, Bd, α, β))
                bench_cutile_zoo(case, "jds", flops, Cd, ctx, n, α2, β2;
                    per_row=1,
                    build=(tm, tn, bnz) -> build_spmm_jds(T; tile_m=tm, tile_n=tn,
                                                          pm=false, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, jds_col_abs, jds_iter_pad,
                                               jds_nz, Bd, α, β))
                # KA JDS baselines
                jds_pm_ka! = spmm_jds_pm_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA jds pm", flops, Cd, ctx; init0=:zero,
                    f0=C -> jds_pm_ka!(C, jds_col, jds_iter, Bd, T(1), T(0);
                                       ndrange=(mm, n)),
                    fab=C -> jds_pm_ka!(C, jds_col, jds_iter, Bd, α2, β2;
                                        ndrange=(mm, n)))
                jds_ka! = spmm_jds_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA jds", flops, Cd, ctx; init0=:zero,
                    f0=C -> jds_ka!(C, jds_col_abs, jds_iter, jds_nz, Bd,
                                    T(1), T(0); ndrange=(mm, n)))
            else
                # cuTile 2-per-row
                bench_cutile_zoo(case, "2pr pm", flops, Cd, ctx, n, α2, β2;
                    per_row=2,
                    build=(tm, tn, bnz) -> build_spmm_2pr(T; tile_m=tm, tile_n=tn,
                                                          pm=true, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, tpr_col, Bd, α, β))
                bench_cutile_zoo(case, "2pr", flops, Cd, ctx, n, α2, β2;
                    per_row=2,
                    build=(tm, tn, bnz) -> build_spmm_2pr(T; tile_m=tm, tile_n=tn,
                                                          pm=false, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, tpr_col, tpr_vals, Bd, α, β))
                # KA 2-per-row baselines
                tpr_pm_ka! = spmm_2pr_pm_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA 2pr pm", flops, Cd, ctx; init0=:zero,
                    f0=C -> tpr_pm_ka!(C, tpr_col2, Bd, T(1), T(0);
                                       ndrange=(mm, n)),
                    fab=C -> tpr_pm_ka!(C, tpr_col2, Bd, α2, β2;
                                        ndrange=(mm, n)))
                tpr_ka! = spmm_2pr_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA 2pr", flops, Cd, ctx; init0=:zero,
                    f0=C -> tpr_ka!(C, tpr_col2, tpr_vals2, Bd, T(1), T(0);
                                    ndrange=(mm, n)))
            end

            # KA CSR row-per-thread (CoolPDLP's spmm_csr!)
            csr_ka! = spmm_csr_ka!(KA.get_backend(Cd))
            bench_impl(case, "KA csr", flops, Cd, ctx; init0=:zero,
                f0=C -> csr_ka!(C, csr.rowptr, csr.colval, csr.nzval, Bd,
                                T(1), T(0); ndrange=(mm, n)))

            # vendor sparse library
            try
                Av = vendor_csr(csr.rowptr, csr.colval, csr.nzval, mm, kk)
                bench_impl(case, VENDOR_NAME, flops, Cd, ctx; init0=:zero,
                    f0=C -> mul!(C, Av, Bd, one(T), zero(T)),
                    fab=C -> mul!(C, Av, Bd, α2, β2))
            catch err
                println("RESULT\t$case\t$VENDOR_NAME\tFAILED\t",
                        replace(first(sprint(showerror, err), 100), "\n" => " "))
            end

            Bd = Cd = nothing
            GC.gc()
            BACKEND == "cuda" && CUDA.reclaim()
        end
    end
    println("SPMM FLOW BENCH DONE ($BACKEND)")
end

main()
