# SpMM bench on the min-cost-flow constraint matrix (TX road network,
# node-arc incidence — the matrix from CoolPDLP2/reactant/benchmarks.jl),
# in every format we have a kernel for:
#
#   A  (nodes×arcs, ~5 nnz/row):  cuTile CSR, cuTile JDS and range+CSR
#                                 (pm/vals), KA jds, KA rc, KA csr, cuSPARSE/rocSPARSE,
#                                 and the hybrid rc + heavy rows ("rch") when
#                                 the instance has rows above SPMM_HEAVY_MAXLEN
#   Aᵀ (arcs×nodes, ≤2 nnz/row):  cuTile CSR, cuTile 2-per-row (pm/vals),
#                                 KA 2pr, KA csr, cuSPARSE/rocSPARSE
#
#   julia --project=. bench/spmm_flow.jl [n...]      (default n = 8, 64, 256)
#   SPMM_MATS=At restricts to one matrix; SPMM_TYPE=Float64 switches the eltype;
#   SPMM_DATA=vision_rnd_05 picks another exported instance (default flow_TX).
#
# Needs bench/data/<SPMM_DATA>{,_t,_zoo}.jls, serialized by the export script
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
include(joinpath(@__DIR__, "spmm_formats.jl"))

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
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
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
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
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

"Is slot id `j` present? With `EXACT2` every row is known to hold two
entries (each arc has a tail and a head), so the `j > 0` test is dropped."
@inline slot_present(j, ::Val{false}) = j > 0
@inline slot_present(j, ::Val{true}) = true

@kernel function spmm_2pr_pm_ka!(c, colidx, b, α, β, ::Val{NB}, ::Val{BT},
                                 ::Val{BETA_NZ}, ::Val{EXACT2}) where {NB, BT, BETA_NZ, EXACT2}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        j1 = colidx[1, i]
        slot_present(j1, Val(EXACT2)) && (s += nb_load(b, j1, b0, Val(NB), Val(BT)))
        j2 = colidx[2, i]
        slot_present(j2, Val(EXACT2)) && (s -= nb_load(b, j2, b0, Val(NB), Val(BT)))
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

@kernel function spmm_2pr_ka!(c, colidx, vals, b, α, β, ::Val{NB}, ::Val{BT},
                              ::Val{BETA_NZ}, ::Val{EXACT2}) where {NB, BT, BETA_NZ, EXACT2}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        j1 = colidx[1, i]
        slot_present(j1, Val(EXACT2)) &&
            (s = muladd(vals[1, i], nb_load(b, j1, b0, Val(NB), Val(BT)), s))
        j2 = colidx[2, i]
        slot_present(j2, Val(EXACT2)) &&
            (s = muladd(vals[2, i], nb_load(b, j2, b0, Val(NB), Val(BT)), s))
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

# Heavy rows of the hybrid range+CSR ("rch") format: pass 1 sums CHUNK-entry
# interleaved slices of a heavy row per thread (chunk c of heavy row h =
# crow[c] takes the entries k ≡ c - cptr[h] mod nchunks(h), so consecutive
# threads read consecutive B rows of the range), pass 2 adds the per-chunk
# partials into C. The partial matrix is laid out like C (nchunks × n, or
# n × nchunks when transposed).
@kernel function heavy_partials_pm_ka!(part, crow, cptr, hlo, hhi, hptr, hids, b,
                                       ::Val{NB}, ::Val{BT}) where {NB, BT}
    c, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        h = crow[c]
        c0 = cptr[h]; nc = cptr[h + 1] - c0
        lo = hlo[h]; rlen = hhi[h] - lo
        p0 = hptr[h]; len = rlen + hptr[h + 1] - p0
        s = zero(Vec{NB, eltype(part)})
        kk = c - c0
        while kk < rlen
            s += nb_load(b, lo + kk, b0, Val(NB), Val(BT))
            kk += nc
        end
        while kk < len
            s -= nb_load(b, hids[p0 + kk - rlen], b0, Val(NB), Val(BT))
            kk += nc
        end
        nb_store!(part, s, c, b0, one(eltype(part)), zero(eltype(part)), Val(BT), Val(false))
    end
end

@kernel function heavy_partials_ka!(part, crow, cptr, hlo, hhi, rvals, hptr, hids, hinvals, b,
                                    ::Val{NB}, ::Val{BT}) where {NB, BT}
    c, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        h = crow[c]
        c0 = cptr[h]; nc = cptr[h + 1] - c0
        lo = hlo[h]; rlen = hhi[h] - lo
        p0 = hptr[h]; len = rlen + hptr[h + 1] - p0
        s = zero(Vec{NB, eltype(part)})
        kk = c - c0
        while kk < rlen
            s = muladd(rvals[lo + kk], nb_load(b, lo + kk, b0, Val(NB), Val(BT)), s)
            kk += nc
        end
        while kk < len
            p = p0 + kk - rlen
            s = muladd(hinvals[p], nb_load(b, hids[p], b0, Val(NB), Val(BT)), s)
            kk += nc
        end
        nb_store!(part, s, c, b0, one(eltype(part)), zero(eltype(part)), Val(BT), Val(false))
    end
end

"C[hrow[h], :] += α · Σ partials of heavy row h (C already holds β·C there)."
@kernel function heavy_reduce_ka!(c, part, hrow, cptr, α, ::Val{NB}, ::Val{BT}) where {NB, BT}
    h, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for cc in cptr[h]:(cptr[h + 1] - Int32(1))
            s += nb_load(part, cc, b0, Val(NB), Val(BT))
        end
        nb_store!(c, s, hrow[h], b0, α, one(eltype(c)), Val(BT), Val(true))
    end
end

"Thread → (row, column block). In the transposed layout the ndrange is
(n ÷ NB, m) so consecutive threads take consecutive rhs columns: the
contiguous direction of the n×k B and n×m C."
@inline ka_rowcol(idx, ::Val{false}) = idx
@inline ka_rowcol(idx, ::Val{true}) = (idx[2], idx[1])
ka_ndrange(m, nblk, bt) = bt ? (nblk, m) : (m, nblk)

# range + CSR (see spmm_zoo_kernels.jl): contiguous column range [lo, hi)
# of value rsign, scattered CSR columns of value -rsign
@kernel function spmm_rc_pm_ka!(c, lo, hi, inptr, inids, b, α, β, rsign, ::Val{NB},
                                ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in lo[i]:(hi[i] - Int32(1))
            s += nb_load(b, a, b0, Val(NB), Val(BT))
        end
        for p in inptr[i]:(inptr[i + 1] - Int32(1))
            s -= nb_load(b, inids[p], b0, Val(NB), Val(BT))
        end
        nb_store!(c, s, i, b0, α * rsign, β, Val(BT), Val(BETA_NZ))
    end
end

@kernel function spmm_rc_ka!(c, lo, hi, rvals, inptr, inids, invals, b, α, β, ::Val{NB},
                             ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in lo[i]:(hi[i] - Int32(1))
            s = muladd(rvals[a], nb_load(b, a, b0, Val(NB), Val(BT)), s)
        end
        for p in inptr[i]:(inptr[i + 1] - Int32(1))
            s = muladd(invals[p], nb_load(b, inids[p], b0, Val(NB), Val(BT)), s)
        end
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

# Hybrid split: rows above HEAVY_MAXLEN nonzeros go to the chunk-parallel
# heavy passes; HEAVY_CHUNK_* is the nonzeros per chunk (per thread for KA,
# per program — a multiple of the heavy tile_k — for cuTile).
const HEAVY_MAXLEN = parse(Int, get(ENV, "SPMM_HEAVY_MAXLEN", "64"))
const HEAVY_CHUNK_KA = parse(Int, get(ENV, "SPMM_HEAVY_CHUNK_KA", "64"))
const HEAVY_CHUNK_CT = parse(Int, get(ENV, "SPMM_HEAVY_CHUNK_CT", "256"))
const HEAVY_TILE_K = 32

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
    data = get(ENV, "SPMM_DATA", "flow_TX")

    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$data.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$(data)_t.jls")))
    zoo = deserialize(joinpath(datadir, "$(data)_zoo.jls"))
    m, k = size(A)
    println("# spmm_flow bench: $BACKEND ($(device_name())), $T, $data: ",
            "A = $m×$k with nnz=$(nnz(A)) (node-arc incidence), n = $ns")
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

    # range + CSR arrays of A, derived from its CSR
    rc = range_csr(At.colptr, At.rowval, At.nzval)
    println("# range+CSR: range sign $(rc.rsign), $(length(rc.inids)) scattered of $(nnz(A)) nnz, ",
            "max range $(maximum(rc.hi .- rc.lo)), max scattered $(maximum(diff(rc.inptr)))")
    rc_lo = GPUArr(rc.lo); rc_hi = GPUArr(rc.hi)
    rc_inptr = GPUArr(rc.inptr); rc_inids = GPUArr(rc.inids)
    rc_rvals = GPUArr(fill(T(rc.rsign), k)); rc_invals = GPUArr(fill(T(-rc.rsign), length(rc.inids)))
    rsign = T(rc.rsign)

    # hybrid: rows longer than HEAVY_MAXLEN peeled out (none on the road matrices)
    rch = split_heavy(rc; maxlen=HEAVY_MAXLEN)
    nheavy = length(rch.heavy.row)
    println("# hybrid rc: $nheavy heavy rows (> $HEAVY_MAXLEN nnz), ",
            "$(sum(rch.heavy.len; init=0)) of $(nnz(A)) nnz")
    if nheavy > 0
        l = rch.light
        l_lo = GPUArr(l.lo); l_hi = GPUArr(l.hi); l_inptr = GPUArr(l.inptr); l_inids = GPUArr(l.inids)
        l_invals = GPUArr(fill(T(-rc.rsign), length(l.inids)))
        hv = rch.heavy
        h_row = GPUArr(hv.row); h_lo = GPUArr(hv.lo); h_hi = GPUArr(hv.hi)
        h_ptr = GPUArr(hv.ptr); h_ids = GPUArr(hv.ids)
        h_invals = GPUArr(fill(T(-rc.rsign), length(hv.ids)))
        ka_chunks = heavy_chunks(hv; chunk=HEAVY_CHUNK_KA)
        ct_chunks = heavy_chunks(hv; chunk=HEAVY_CHUNK_CT)
        ka_crow = GPUArr(ka_chunks.crow); ka_cptr = GPUArr(ka_chunks.cptr)
        ct_crow = GPUArr(ct_chunks.crow); ct_cptr = GPUArr(ct_chunks.cptr)
    end

    # 2-per-row arrays of Aᵀ (the 2×m slot matrices, shared by cuTile and KA)
    tpr_col2 = GPUArr(zoo.tpr_colidx)
    tpr_vals2 = GPUArr(repeat(T[1, -1], 1, size(zoo.tpr_colidx, 2)))
    @assert size(zoo.tpr_colidx, 2) == k && zoo.tpr_ncols == m
    tpr_exact2 = all(>(0), zoo.tpr_colidx)   # every arc has a tail and a head

    mats = split(get(ENV, "SPMM_MATS", "A,At"), ",")   # e.g. SPMM_MATS=At
    for (mat, csr, Amat) in (("A", csrA, A), ("At", csrAt, At))
        mat in mats || continue
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
                        build=((tm, tn, nw), bnz) -> build_spmm_jds(T; tile_m=tm,
                            tile_n=tn, pm=true, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, jds_col, jds_iter, Bx, α, β))
                    bench_cutile(case, "jds$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=1),
                        build=((tm, tn, nw), bnz) -> build_spmm_jds(T; tile_m=tm,
                            tile_n=tn, pm=false, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, jds_col_abs, jds_iter,
                                                   jds_nz, Bx, α, β))
                    # KA JDS baselines, swept over NB columns per thread
                    jds_pm_ka! = spmm_jds_pm_ka!(KA.get_backend(Cd))
                    jds_ka! = spmm_jds_ka!(KA.get_backend(Cd))
                    for nb in (1, 4, 8)
                        n % nb == 0 || continue
                        grid = ka_ndrange(mm, n ÷ nb, bt)
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
                    # cuTile range + CSR
                    bench_cutile(case, "rc pm$lay", flops, Cd, ctx, α2, β2;
                        cands=rc_tile_candidates(n),
                        build=((tm, tn, tk, nw), bnz) -> build_spmm_rc(T; tile_m=tm,
                            tile_n=tn, tile_k=tk, pm=true, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, rc_lo, rc_hi, rc_inptr, rc_inids,
                                                   rsign, Bx, α, β))
                    bench_cutile(case, "rc$lay", flops, Cd, ctx, α2, β2;
                        cands=rc_tile_candidates(n),
                        build=((tm, tn, tk, nw), bnz) -> build_spmm_rc(T; tile_m=tm,
                            tile_n=tn, tile_k=tk, pm=false, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, rc_lo, rc_hi, rc_rvals, rc_inptr,
                                                   rc_inids, rc_invals, Bx, α, β))
                    # KA range + CSR
                    rc_pm_ka! = spmm_rc_pm_ka!(KA.get_backend(Cd))
                    rc_ka! = spmm_rc_ka!(KA.get_backend(Cd))
                    for nb in (1, 4, 8)
                        n % nb == 0 || continue
                        grid = ka_ndrange(mm, n ÷ nb, bt)
                        bench_impl(case, "KA rc pm[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> rc_pm_ka!(C, rc_lo, rc_hi, rc_inptr, rc_inids, Bx,
                                              T(1), T(0), rsign, Val(nb), Val(bt),
                                              Val(false); ndrange=grid),
                            fab=C -> rc_pm_ka!(C, rc_lo, rc_hi, rc_inptr, rc_inids, Bx,
                                               α2, β2, rsign, Val(nb), Val(bt),
                                               Val(true); ndrange=grid))
                        bench_impl(case, "KA rc[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> rc_ka!(C, rc_lo, rc_hi, rc_rvals, rc_inptr, rc_inids,
                                           rc_invals, Bx, T(1), T(0), Val(nb), Val(bt),
                                           Val(false); ndrange=grid))
                    end

                    # hybrid rc + heavy rows ("rch"): rc kernels on the light
                    # arrays, then the chunk partials and the reduce pass
                    if nheavy > 0
                        ct_part = GPUArr(zeros(T, bt ? (n, length(ct_chunks.crow)) :
                                                       (length(ct_chunks.crow), n)))
                        ka_part = GPUArr(zeros(T, bt ? (n, length(ka_chunks.crow)) :
                                                       (length(ka_chunks.crow), n)))
                        bench_cutile(case, "rch pm$lay", flops, Cd, ctx, α2, β2;
                            cands=rc_tile_candidates(n),
                            build=((tm, tn, tk, nw), bnz) -> begin
                                light! = build_spmm_rc(T; tile_m=tm, tile_n=tn, tile_k=tk,
                                                       pm=true, beta_nz=bnz, bt, num_warps=nw)
                                part!, red! = build_cached(("heavy", T, tn, nw, true, bt),
                                    () -> build_spmm_heavy(T; tile_n=tn, tile_k=HEAVY_TILE_K,
                                        chunk=HEAVY_CHUNK_CT, pm=true, bt, num_warps=nw))
                                (C, part, B, α, β) -> begin
                                    light!(C, l_lo, l_hi, l_inptr, l_inids, rsign, B, α, β)
                                    part!(part, ct_crow, ct_cptr, h_lo, h_hi, h_ptr, h_ids, rsign, B)
                                    red!(C, part, h_row, ct_cptr, α)
                                end
                            end,
                            launch=(f!, C, α, β) -> f!(C, ct_part, Bx, α, β))
                        bench_cutile(case, "rch$lay", flops, Cd, ctx, α2, β2;
                            cands=rc_tile_candidates(n),
                            build=((tm, tn, tk, nw), bnz) -> begin
                                light! = build_spmm_rc(T; tile_m=tm, tile_n=tn, tile_k=tk,
                                                       pm=false, beta_nz=bnz, bt, num_warps=nw)
                                part!, red! = build_cached(("heavy", T, tn, nw, false, bt),
                                    () -> build_spmm_heavy(T; tile_n=tn, tile_k=HEAVY_TILE_K,
                                        chunk=HEAVY_CHUNK_CT, pm=false, bt, num_warps=nw))
                                (C, part, B, α, β) -> begin
                                    light!(C, l_lo, l_hi, rc_rvals, l_inptr, l_inids, l_invals, B, α, β)
                                    part!(part, ct_crow, ct_cptr, h_lo, h_hi, rc_rvals, h_ptr,
                                          h_ids, h_invals, B)
                                    red!(C, part, h_row, ct_cptr, α)
                                end
                            end,
                            launch=(f!, C, α, β) -> f!(C, ct_part, Bx, α, β))
                        hp_pm_ka! = heavy_partials_pm_ka!(KA.get_backend(Cd))
                        hp_ka! = heavy_partials_ka!(KA.get_backend(Cd))
                        hr_ka! = heavy_reduce_ka!(KA.get_backend(Cd))
                        for nb in (1, 4, 8)
                            n % nb == 0 || continue
                            grid = ka_ndrange(mm, n ÷ nb, bt)
                            pgrid = ka_ndrange(length(ka_chunks.crow), n ÷ nb, bt)
                            rgrid = ka_ndrange(nheavy, n ÷ nb, bt)
                            rch_pm! = (C, α, β, bnz) -> begin
                                rc_pm_ka!(C, l_lo, l_hi, l_inptr, l_inids, Bx, α, β, rsign,
                                          Val(nb), Val(bt), bnz; ndrange=grid)
                                hp_pm_ka!(ka_part, ka_crow, ka_cptr, h_lo, h_hi, h_ptr, h_ids, Bx,
                                          Val(nb), Val(bt); ndrange=pgrid)
                                hr_ka!(C, ka_part, h_row, ka_cptr, α * rsign, Val(nb), Val(bt);
                                       ndrange=rgrid)
                            end
                            bench_impl(case, "KA rch pm[nb=$nb$lay]", flops, Cd, ctx;
                                f0=C -> rch_pm!(C, T(1), T(0), Val(false)),
                                fab=C -> rch_pm!(C, α2, β2, Val(true)))
                            bench_impl(case, "KA rch[nb=$nb$lay]", flops, Cd, ctx;
                                f0=C -> begin
                                    rc_ka!(C, l_lo, l_hi, rc_rvals, l_inptr, l_inids, l_invals, Bx,
                                           T(1), T(0), Val(nb), Val(bt), Val(false); ndrange=grid)
                                    hp_ka!(ka_part, ka_crow, ka_cptr, h_lo, h_hi, rc_rvals, h_ptr,
                                           h_ids, h_invals, Bx, Val(nb), Val(bt); ndrange=pgrid)
                                    hr_ka!(C, ka_part, h_row, ka_cptr, T(1), Val(nb), Val(bt);
                                           ndrange=rgrid)
                                end)
                        end
                        ct_part = ka_part = nothing
                    end
                else
                    # cuTile 2-per-row
                    bench_cutile(case, "2pr pm$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=2),
                        build=((tm, tn, nw), bnz) -> build_spmm_2pr(T; tile_m=tm,
                            tile_n=tn, pm=true, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, tpr_col2, Bx, α, β))
                    bench_cutile(case, "2pr$lay", flops, Cd, ctx, α2, β2;
                        cands=zoo_tile_candidates(n; per_row=2),
                        build=((tm, tn, nw), bnz) -> build_spmm_2pr(T; tile_m=tm,
                            tile_n=tn, pm=false, beta_nz=bnz, bt, num_warps=nw),
                        launch=(f!, C, α, β) -> f!(C, tpr_col2, tpr_vals2, Bx, α, β))
                    # exactly-2-per-row variants: no id-0 checks on the B gather
                    # (labels "… x2"); valid because every arc has both endpoints
                    if tpr_exact2
                        bench_cutile(case, "2pr pm x2$lay", flops, Cd, ctx, α2, β2;
                            cands=zoo_tile_candidates(n; per_row=2),
                            build=((tm, tn, nw), bnz) -> build_spmm_2pr(T; tile_m=tm,
                                tile_n=tn, pm=true, beta_nz=bnz, bt, num_warps=nw,
                                exact2=true),
                            launch=(f!, C, α, β) -> f!(C, tpr_col2, Bx, α, β))
                        bench_cutile(case, "2pr x2$lay", flops, Cd, ctx, α2, β2;
                            cands=zoo_tile_candidates(n; per_row=2),
                            build=((tm, tn, nw), bnz) -> build_spmm_2pr(T; tile_m=tm,
                                tile_n=tn, pm=false, beta_nz=bnz, bt, num_warps=nw,
                                exact2=true),
                            launch=(f!, C, α, β) -> f!(C, tpr_col2, tpr_vals2, Bx, α, β))
                    end
                    # KA 2-per-row baselines, swept over NB columns per thread
                    tpr_pm_ka! = spmm_2pr_pm_ka!(KA.get_backend(Cd))
                    tpr_ka! = spmm_2pr_ka!(KA.get_backend(Cd))
                    for nb in (1, 4, 8)
                        n % nb == 0 || continue
                        grid = ka_ndrange(mm, n ÷ nb, bt)
                        bench_impl(case, "KA 2pr pm[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> tpr_pm_ka!(C, tpr_col2, Bx, T(1), T(0), Val(nb),
                                               Val(bt), Val(false), Val(false); ndrange=grid),
                            fab=C -> tpr_pm_ka!(C, tpr_col2, Bx, α2, β2, Val(nb),
                                                Val(bt), Val(true), Val(false); ndrange=grid))
                        bench_impl(case, "KA 2pr[nb=$nb$lay]", flops, Cd, ctx;
                            f0=C -> tpr_ka!(C, tpr_col2, tpr_vals2, Bx, T(1), T(0),
                                            Val(nb), Val(bt), Val(false), Val(false);
                                            ndrange=grid))
                        tpr_exact2 || continue
                        bench_impl(case, "KA 2pr pm[nb=$nb x2$lay]", flops, Cd, ctx;
                            f0=C -> tpr_pm_ka!(C, tpr_col2, Bx, T(1), T(0), Val(nb),
                                               Val(bt), Val(false), Val(true); ndrange=grid))
                        bench_impl(case, "KA 2pr[nb=$nb x2$lay]", flops, Cd, ctx;
                            f0=C -> tpr_ka!(C, tpr_col2, tpr_vals2, Bx, T(1), T(0),
                                            Val(nb), Val(bt), Val(false), Val(true);
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

                # free eagerly: the next layout's set must not coexist with
                # this one (5 C-sized arrays each; OOM on the 23M-row Aᵀ)
                BACKEND == "cuda" && foreach(CUDA.unsafe_free!, (Bx, Cd, C0, Cref, Cref2))
                Bx = Cd = C0 = Cref = Cref2 = ctx = nothing
                GC.gc()
                BACKEND == "cuda" && CUDA.reclaim()
            end
        end
    end
    println("SPMM FLOW BENCH DONE ($BACKEND)")
end

main()
