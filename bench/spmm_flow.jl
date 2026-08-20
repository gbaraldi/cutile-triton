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
# The zoo kernels (cuTile and KA) run in the standard layout and a fully
# transposed one — B as n×k and C as n×m, rhs columns contiguous (labels
# "… t"); the KA kernels additionally sweep NB rhs columns per thread
# (labels "[nb=…]").
# Environment knobs as in bench/spmm_csr.jl: SPMM_BACKEND, SPMM_TYPE
# (default Float32 here, matching CoolPDLP).

using LinearAlgebra, SparseArrays, Random, Printf, Serialization
using SIMD

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_csr_kernels.jl"))
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

# --- KA baselines: SpMM versions of the matrix_zoo kernels; the csr baseline
# --- comes from spmm_common.jl.
#
# Each thread covers one row × NB rhs columns with a SIMD.Vec accumulator, so
# colidx/vals are read once per NB columns instead of once per column — at
# nb=1 (the original one-thread-per-element shape) that n-fold redundant
# metadata traffic is what makes the vals variants ~2× slower than pm
# (spmm_flow_results.md). On the GPU NVPTX scalarizes the Vec arithmetic, so
# this matches hand-unrolled tuples; with the transposed-B layout (BT: b is
# n×k, the NB columns contiguous) the lanes of nb_load can fuse into real
# vector loads.

"NB-column block of B at column id `col` as a SIMD vector; `BT` picks the
b[n×k] transposed layout. A top-level function so the ntuple closure captures
only arguments — a captured reassigned kernel local would be boxed, which GPU
compilation can't take."
@inline nb_load(b, col, b0, ::Val{NB}, ::Val{true}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[b0 + q, col]), Val(NB)))
@inline nb_load(b, col, b0, ::Val{NB}, ::Val{false}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[col, b0 + q]), Val(NB)))

"α/β-update row i, columns b0+1:b0+NB of C with the accumulator lanes
(column i of the transposed n×m C when `BT`). `BETA_NZ = false` is the
β = 0 specialization that never reads C — what MinimumCostFlows' `mul!`
gets from its `Zero()` dispatch."
@inline function nb_store!(c, s::Vec{NB}, i, b0, α, β,
                           ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    @inbounds for q in 1:NB
        v = α * s[q]
        if BT
            c[b0 + q, i] = BETA_NZ ? v + β * c[b0 + q, i] : v
        else
            c[i, b0 + q] = BETA_NZ ? v + β * c[i, b0 + q] : v
        end
    end
end

@kernel function spmm_jds_pm_ka!(c, colidx, iterptr, b, α, β, ::Val{NB},
                                 ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = @index(Global, NTuple)
    b0 = (blk - 1) * NB
    T = eltype(c)
    @inbounds begin
        s = zero(Vec{NB, T})
        j = 1
        ptr = i
        while j < length(iterptr) && ptr < iterptr[j + 1]
            k = colidx[ptr]
            s = muladd(flipsign(one(T), k),
                       nb_load(b, abs(k), b0, Val(NB), Val(BT)), s)
            ptr = i + iterptr[j += 1] - 1
        end
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

@kernel function spmm_jds_ka!(c, colidx, iterptr, vals, b, α, β, ::Val{NB},
                              ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = @index(Global, NTuple)
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        j = 1
        ptr = i
        while j < length(iterptr) && ptr < iterptr[j + 1]
            s = muladd(vals[ptr],
                       nb_load(b, colidx[ptr], b0, Val(NB), Val(BT)), s)
            ptr = i + iterptr[j += 1] - 1
        end
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

@kernel function spmm_2pr_pm_ka!(c, colidx, b, α, β, ::Val{NB},
                                 ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = @index(Global, NTuple)
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        j1 = colidx[1, i]
        j1 > 0 && (s += nb_load(b, j1, b0, Val(NB), Val(BT)))
        j2 = colidx[2, i]
        j2 > 0 && (s -= nb_load(b, j2, b0, Val(NB), Val(BT)))
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

@kernel function spmm_2pr_ka!(c, colidx, vals, b, α, β, ::Val{NB},
                              ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = @index(Global, NTuple)
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        j1 = colidx[1, i]
        j1 > 0 && (s = muladd(vals[1, i], nb_load(b, j1, b0, Val(NB), Val(BT)), s))
        j2 = colidx[2, i]
        j2 > 0 && (s = muladd(vals[2, i], nb_load(b, j2, b0, Val(NB), Val(BT)), s))
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

# ---------------------------------------------------------------------------

"""
Verify and time one implementation. `f0(Cd)` runs the β=0 path in place,
`fab(Cd)` the general α/β path (or `nothing` to skip); `init0` is what Cd
must hold before `f0` (NaN poison for kernels that never read C, 0 for the
KA-csr/vendor paths that do). Returns the β=0 best time (or nothing).
"""
function bench_impl(case, label, flops, Cd, ctx; f0, fab=nothing, init0=NaN)
    (; Cref, Cref2, C0, rtol) = ctx
    fill!(Cd, eltype(Cd)(init0))
    f0(Cd)
    device_sync()
    check(label, Cd, Cref; rtol) || return nothing
    t = timeit(() -> f0(Cd))
    result_row(case, label, t, flops)
    if fab !== nothing
        copyto!(Cd, C0)
        fab(Cd)
        device_sync()
        check("$label αβ", Cd, Cref2; rtol)
        tab = timeit(() -> fab(Cd))
        result_row(case, "$label αβ", tab, flops)
    end
    flush(stdout)
    return t
end

# The same kernel specialization is requested repeatedly across the n-loop
# and the A/Aᵀ loop (the candidate lists largely coincide); cache the
# launchers — and any compile failure — so each Triton compile runs once.
const BUILD_CACHE = Dict{Any, Any}()
build_cached(key, build) = get!(() -> try build() catch err; err end,
                                BUILD_CACHE, key)

"""
Tune a cuTile kernel over the `cands` tile configs: `build(cfg, beta_nz)`
returns a launcher, `launch(spmm!, C, α, β)` runs it. The best β=0 config is
then rebuilt with `beta_nz=true` and verified/timed on the general α/β path.
"""
function bench_cutile(case, name, flops, Cd, ctx, α2, β2; cands, build, launch)
    best = nothing
    for cfg in cands
        label = "cuTile $name[$(join(cfg, "×"))]"
        spmm! = build_cached((name, ctx.T, cfg, false), () -> build(cfg, false))
        if spmm! isa Exception
            fail_row(case, label, spmm!)
            continue
        end
        t = bench_impl(case, label, flops, Cd, ctx;
                       f0=C -> launch(spmm!, C, 1, 0))
        t === nothing && continue
        (best === nothing || t < best[1]) && (best = (t, cfg))
    end
    if best !== nothing
        (_, cfg) = best
        label = "cuTile $name[$(join(cfg, "×"))]"
        spmm_ab! = build_cached((name, ctx.T, cfg, true), () -> build(cfg, true))
        spmm_ab! isa Exception && return fail_row(case, "$label αβ", spmm_ab!)
        copyto!(Cd, ctx.C0)                     # reads C: needs it initialized
        launch(spmm_ab!, Cd, α2, β2)
        device_sync()
        check("$label αβ", Cd, ctx.Cref2; rtol=ctx.rtol)
        t = timeit(() -> launch(spmm_ab!, Cd, α2, β2))
        result_row(case, "$label best αβ", t, flops)
        flush(stdout)
    end
end

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = get(ENV, "SPMM_TYPE", "Float32") == "Float64" ? Float64 : Float32
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

    # JDS arrays of A (identity row map)
    jds_col = GPUArr(zoo.jds_colidx)
    jds_col_abs = GPUArr(abs.(zoo.jds_colidx))
    jds_nz = GPUArr(flipsign.(one(T), zoo.jds_colidx))
    jds_iter = GPUArr(zoo.jds_iterptr)
    @assert zoo.jds_nrows == m && zoo.jds_ncols == k

    # 2-per-row arrays of Aᵀ (the 2×m slot matrices, shared by cuTile and KA)
    tpr_col2 = GPUArr(zoo.tpr_colidx)
    tpr_vals2 = GPUArr(repeat(T[1, -1], 1, size(zoo.tpr_colidx, 2)))
    @assert size(zoo.tpr_colidx, 2) == k && zoo.tpr_ncols == m

    for (mat, csr, Amat) in (("A", csrA, A), ("At", csrAt, At))
        mm, kk = size(Amat)
        for n in ns
            case = "flow $mat n=$n $T"
            flops = 2.0 * nnz(Amat) * n
            Bh = rand(rng, T, kk, n)
            C0h = rand(rng, T, mm, n)
            Crefh = Amat * Bh

            # Standard layout, then the fully transposed one (B as n×kk, C as
            # n×mm — rhs columns contiguous; labels "… t"). Device arrays are
            # rebuilt per layout to bound device memory; the references live
            # on the device so checks run there instead of downloading the
            # full C per candidate. csr/vendor baselines run standard-only.
            for (bt, lay) in ((false, ""), (true, " t"))
                Bx = GPUArr(bt ? permutedims(Bh) : Bh)
                C0 = GPUArr(bt ? permutedims(C0h) : C0h)
                Cref = GPUArr(bt ? permutedims(Crefh) : Crefh)
                Cref2 = α2 .* Cref .+ β2 .* C0
                Cd = similar(Cref)
                ctx = (; T, Cref, Cref2, C0, rtol)

                if !bt
                    # cuTile CSR
                    bench_cutile(case, "csr", flops, Cd, ctx, α2, β2;
                        cands=csr_tile_candidates(n),
                        build=((tm, tn, tk), bnz) -> build_spmm(T; tile_m=tm,
                            tile_n=tn, tile_k=tk, beta_nz=bnz),
                        launch=(f!, C, α, β) -> f!(C, csr.rowptr, csr.colval,
                                                   csr.nzval, Bx, α, β))
                end

                if mat == "A"
                    # cuTile JDS
                    bench_cutile(case, "jds pm$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=1),
                        build=((tm, tn), bnz) -> build_spmm_jds(T; tile_m=tm,
                            tile_n=tn, pm=true, beta_nz=bnz, bt),
                        launch=(f!, C, α, β) -> f!(C, jds_col, jds_iter, Bx, α, β))
                    bench_cutile(case, "jds$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=1),
                        build=((tm, tn), bnz) -> build_spmm_jds(T; tile_m=tm,
                            tile_n=tn, pm=false, beta_nz=bnz, bt),
                        launch=(f!, C, α, β) -> f!(C, jds_col_abs, jds_iter,
                                                   jds_nz, Bx, α, β))
                    # KA JDS baselines, swept over NB columns per thread
                    jds_pm_ka! = spmm_jds_pm_ka!(KA.get_backend(Cd))
                    jds_ka! = spmm_jds_ka!(KA.get_backend(Cd))
                    for nb in (1, 4, 8)
                        n % nb == 0 || continue
                        grid = (mm, n ÷ nb)
                        bench_impl(case, "KA jds pm[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> jds_pm_ka!(C, jds_col, jds_iter, Bx, T(1),
                                               T(0), Val(nb), Val(bt), Val(false);
                                               ndrange=grid),
                            fab=C -> jds_pm_ka!(C, jds_col, jds_iter, Bx, α2, β2,
                                                Val(nb), Val(bt), Val(true);
                                                ndrange=grid))
                        bench_impl(case, "KA jds[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> jds_ka!(C, jds_col_abs, jds_iter, jds_nz, Bx,
                                            T(1), T(0), Val(nb), Val(bt), Val(false);
                                            ndrange=grid))
                    end
                else
                    # cuTile 2-per-row
                    bench_cutile(case, "2pr pm$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=2),
                        build=((tm, tn), bnz) -> build_spmm_2pr(T; tile_m=tm,
                            tile_n=tn, pm=true, beta_nz=bnz, bt),
                        launch=(f!, C, α, β) -> f!(C, tpr_col2, Bx, α, β))
                    bench_cutile(case, "2pr$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=2),
                        build=((tm, tn), bnz) -> build_spmm_2pr(T; tile_m=tm,
                            tile_n=tn, pm=false, beta_nz=bnz, bt),
                        launch=(f!, C, α, β) -> f!(C, tpr_col2, tpr_vals2, Bx, α, β))
                    # KA 2-per-row baselines, swept over NB columns per thread
                    tpr_pm_ka! = spmm_2pr_pm_ka!(KA.get_backend(Cd))
                    tpr_ka! = spmm_2pr_ka!(KA.get_backend(Cd))
                    for nb in (1, 4, 8)
                        n % nb == 0 || continue
                        grid = (mm, n ÷ nb)
                        bench_impl(case, "KA 2pr pm[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> tpr_pm_ka!(C, tpr_col2, Bx, T(1), T(0), Val(nb),
                                               Val(bt), Val(false); ndrange=grid),
                            fab=C -> tpr_pm_ka!(C, tpr_col2, Bx, α2, β2, Val(nb),
                                                Val(bt), Val(true); ndrange=grid))
                        bench_impl(case, "KA 2pr[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> tpr_ka!(C, tpr_col2, tpr_vals2, Bx, T(1), T(0),
                                            Val(nb), Val(bt), Val(false);
                                            ndrange=grid))
                    end
                end

                if !bt
                    # KA CSR row-per-thread (CoolPDLP's spmm_csr!)
                    csr_ka! = spmm_csr_ka!(KA.get_backend(Cd))
                    bench_impl(case, "KA csr", flops, Cd, ctx; init0=0,
                        f0=C -> csr_ka!(C, csr.rowptr, csr.colval, csr.nzval, Bx,
                                        T(1), T(0); ndrange=(mm, n)))

                    # vendor sparse library
                    try
                        Av = vendor_csr(csr.rowptr, csr.colval, csr.nzval, mm, kk)
                        bench_impl(case, VENDOR_NAME, flops, Cd, ctx; init0=0,
                            f0=C -> mul!(C, Av, Bx, one(T), zero(T)),
                            fab=C -> mul!(C, Av, Bx, α2, β2))
                    catch err
                        fail_row(case, VENDOR_NAME, err)
                    end
                end

                Bx = Cd = C0 = Cref = Cref2 = ctx = nothing
                GC.gc()
                BACKEND == "cuda" && CUDA.reclaim()
            end
        end
    end
    println("SPMM FLOW BENCH DONE ($BACKEND)")
end

main()
