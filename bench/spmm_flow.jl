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

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_csr_kernels.jl"))
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

# --- KA baselines: SpMM versions of the matrix_zoo kernels (one thread per
# --- (row, rhs column)); the csr baseline comes from spmm_common.jl ---------

@kernel function spmm_jds_pm_ka!(c, colidx, iterptr, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(c))
        j = 1
        ptr = i
        while j < length(iterptr) && ptr < iterptr[j + 1]
            k = colidx[ptr]
            s += flipsign(b[abs(k), batch_idx], k)
            ptr = i + iterptr[j += 1] - 1
        end
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

@kernel function spmm_jds_ka!(c, colidx, iterptr, vals, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(c))
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
    @inbounds begin
        s = zero(eltype(c))
        j = colidx[1, i]
        j > 0 && (s += b[j, batch_idx])
        j = colidx[2, i]
        j > 0 && (s -= b[j, batch_idx])
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

@kernel function spmm_2pr_ka!(c, colidx, vals, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(c))
        j = colidx[1, i]
        j > 0 && (s += vals[1, i] * b[j, batch_idx])
        j = colidx[2, i]
        j > 0 && (s += vals[2, i] * b[j, batch_idx])
        c[i, batch_idx] = α * s + β * c[i, batch_idx]
    end
end

# ---------------------------------------------------------------------------

"""
Verify and time one implementation. `f0(Cd)` runs the β=0 path in place,
`fab(Cd)` the general α/β path (or `nothing` to skip); `init0` is what Cd
must hold before `f0` (NaN poison for kernels that never read C, 0 for the
KA/vendor paths that do). Returns the β=0 best time (or nothing).
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
            Bd = GPUArr(Bh)
            # references live on the device: checks then run there instead of
            # downloading the full C per candidate
            C0 = GPUArr(rand(rng, T, mm, n))
            Cref = GPUArr(Amat * Bh)
            Cref2 = α2 .* Cref .+ β2 .* C0
            Cd = GPUArr{T}(undef, mm, n)
            ctx = (; T, Cref, Cref2, C0, rtol)

            # cuTile CSR
            bench_cutile(case, "csr", flops, Cd, ctx, α2, β2;
                cands=csr_tile_candidates(n),
                build=((tm, tn, tk), bnz) -> build_spmm(T; tile_m=tm, tile_n=tn,
                                                        tile_k=tk, beta_nz=bnz),
                launch=(f!, C, α, β) -> f!(C, csr.rowptr, csr.colval, csr.nzval,
                                           Bd, α, β))

            if mat == "A"
                # cuTile JDS
                bench_cutile(case, "jds pm", flops, Cd, ctx, α2, β2;
                    cands=zoo_tile_candidates(n; per_row=1),
                    build=((tm, tn), bnz) -> build_spmm_jds(T; tile_m=tm, tile_n=tn,
                                                            pm=true, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, jds_col, jds_iter, Bd, α, β))
                bench_cutile(case, "jds", flops, Cd, ctx, α2, β2;
                    cands=zoo_tile_candidates(n; per_row=1),
                    build=((tm, tn), bnz) -> build_spmm_jds(T; tile_m=tm, tile_n=tn,
                                                            pm=false, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, jds_col_abs, jds_iter,
                                               jds_nz, Bd, α, β))
                # KA JDS baselines
                jds_pm_ka! = spmm_jds_pm_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA jds pm", flops, Cd, ctx; init0=0,
                    f0=C -> jds_pm_ka!(C, jds_col, jds_iter, Bd, T(1), T(0);
                                       ndrange=(mm, n)),
                    fab=C -> jds_pm_ka!(C, jds_col, jds_iter, Bd, α2, β2;
                                        ndrange=(mm, n)))
                jds_ka! = spmm_jds_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA jds", flops, Cd, ctx; init0=0,
                    f0=C -> jds_ka!(C, jds_col_abs, jds_iter, jds_nz, Bd,
                                    T(1), T(0); ndrange=(mm, n)))
            else
                # cuTile 2-per-row
                bench_cutile(case, "2pr pm", flops, Cd, ctx, α2, β2;
                    cands=zoo_tile_candidates(n; per_row=2),
                    build=((tm, tn), bnz) -> build_spmm_2pr(T; tile_m=tm, tile_n=tn,
                                                            pm=true, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, tpr_col2, Bd, α, β))
                bench_cutile(case, "2pr", flops, Cd, ctx, α2, β2;
                    cands=zoo_tile_candidates(n; per_row=2),
                    build=((tm, tn), bnz) -> build_spmm_2pr(T; tile_m=tm, tile_n=tn,
                                                            pm=false, beta_nz=bnz),
                    launch=(f!, C, α, β) -> f!(C, tpr_col2, tpr_vals2, Bd, α, β))
                # KA 2-per-row baselines
                tpr_pm_ka! = spmm_2pr_pm_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA 2pr pm", flops, Cd, ctx; init0=0,
                    f0=C -> tpr_pm_ka!(C, tpr_col2, Bd, T(1), T(0);
                                       ndrange=(mm, n)),
                    fab=C -> tpr_pm_ka!(C, tpr_col2, Bd, α2, β2;
                                        ndrange=(mm, n)))
                tpr_ka! = spmm_2pr_ka!(KA.get_backend(Cd))
                bench_impl(case, "KA 2pr", flops, Cd, ctx; init0=0,
                    f0=C -> tpr_ka!(C, tpr_col2, tpr_vals2, Bd, T(1), T(0);
                                    ndrange=(mm, n)))
            end

            # KA CSR row-per-thread (CoolPDLP's spmm_csr!)
            csr_ka! = spmm_csr_ka!(KA.get_backend(Cd))
            bench_impl(case, "KA csr", flops, Cd, ctx; init0=0,
                f0=C -> csr_ka!(C, csr.rowptr, csr.colval, csr.nzval, Bd,
                                T(1), T(0); ndrange=(mm, n)))

            # vendor sparse library
            try
                Av = vendor_csr(csr.rowptr, csr.colval, csr.nzval, mm, kk)
                bench_impl(case, VENDOR_NAME, flops, Cd, ctx; init0=0,
                    f0=C -> mul!(C, Av, Bd, one(T), zero(T)),
                    fab=C -> mul!(C, Av, Bd, α2, β2))
            catch err
                fail_row(case, VENDOR_NAME, err)
            end

            Bd = Cd = C0 = Cref = Cref2 = ctx = nothing
            GC.gc()
            BACKEND == "cuda" && CUDA.reclaim()
        end
    end
    println("SPMM FLOW BENCH DONE ($BACKEND)")
end

main()
