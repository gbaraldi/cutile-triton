# Does collapsing rc's lo/hi into a single rptr[m+1] array matter?
#
# The ranges of the incidence matrix partition the columns (every arc has
# exactly one head), so in node order they chain — hi[i] == lo[i+1] — and the
# two bounds arrays collapse into one pointer array (= the adjacency matrix's
# CSC colptr). The exported A is in JDS (length-sorted) row order where the
# ranges don't chain, so this benches the rc kernels in three variants:
#
#   lo/hi jds     the current format on the matrix as exported (baseline)
#   lo/hi node    two arrays, rows permuted to node order — isolates the
#                 row-order effect from the array-merge effect
#   rptr node     the single collapsed array, same node order
#
# each as KA (row-per-thread, nb ∈ {1,4,8}) and cuTile (tile sweep), pm and
# vals. The cuTile rptr kernels live here (rp_accumulate/build_spmm_rp):
# rc_accumulate with the range bounds gathered from one array; rows past m
# read rptr[m+1] and a 0-padded rptr[m+2], so rlen goes negative and every
# range/scatter mask is false, as with the 0-padded lo/hi.
#
#   julia --project=. bench/spmm_rptr.jl [n...]      (default n = 8, 64, 256)
#   SPMM_DATA / SPMM_BACKEND / SPMM_TYPE as in bench/spmm_flow.jl.
#
# β=0 path only (the headline configuration); both layouts.

using LinearAlgebra, SparseArrays, Random, Printf, Serialization
using SIMD

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))
include(joinpath(@__DIR__, "spmm_formats.jl"))

# --- KA kernels: nb_load / nb_store! / ka_rowcol / spmm_rc{,_pm}_ka! as in
# --- bench/spmm_flow.jl, plus the single-array rp variants.

@inline nb_load(b, col, b0, ::Val{NB}, ::Val{true}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[b0 + q, col]), Val(NB)))
@inline nb_load(b, col, b0, ::Val{NB}, ::Val{false}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[col, b0 + q]), Val(NB)))

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

@inline ka_rowcol(idx, ::Val{false}) = idx
@inline ka_rowcol(idx, ::Val{true}) = (idx[2], idx[1])
ka_ndrange(m, nblk, bt) = bt ? (nblk, m) : (m, nblk)

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

@kernel function spmm_rp_pm_ka!(c, rptr, inptr, inids, b, α, β, rsign, ::Val{NB},
                                ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in rptr[i]:(rptr[i + 1] - Int32(1))
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

@kernel function spmm_rp_ka!(c, rptr, rvals, inptr, inids, invals, b, α, β, ::Val{NB},
                             ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in rptr[i]:(rptr[i + 1] - Int32(1))
            s = muladd(rvals[a], nb_load(b, a, b0, Val(NB), Val(BT)), s)
        end
        for p in inptr[i]:(inptr[i + 1] - Int32(1))
            s = muladd(invals[p], nb_load(b, inids[p], b0, Val(NB), Val(BT)), s)
        end
        nb_store!(c, s, i, b0, α, β, Val(BT), Val(BETA_NZ))
    end
end

# --- cuTile rptr kernels: rc_accumulate with the range bounds from rptr ----

@inline function rp_accumulate(weights, rptr, inptr, inids, B, bm, bn, BT::Bool,
                               TILE_M::Int, TILE_N::Int, TILE_K::Int, ::Type{T}) where {T}
    rows = tile_span(bm, TILE_M)
    ncols3 = reshape(tile_span(bn, TILE_N), (1, 1, TILE_N))
    ks = reshape(ct.arange(TILE_K) .- Int32(1), (1, TILE_K))
    lo1 = ct.gather(rptr, rows)
    rlen1 = ct.gather(rptr, rows .+ Int32(1)) .- lo1       # rows past m: rlen ≤ 0
    p01 = ct.gather(inptr, rows)
    slen1 = ct.gather(inptr, rows .+ Int32(1)) .- p01
    nk = cld(maximum(rlen1 .+ slen1), Int32(TILE_K))
    los = reshape(lo1, (TILE_M, 1))
    rlen = reshape(rlen1, (TILE_M, 1))
    p0 = reshape(p01, (TILE_M, 1))
    slen = reshape(slen1, (TILE_M, 1))
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nk
        kk = ((t - Int32(1)) * Int32(TILE_K)) .+ ks            # (1, TILE_K)
        inrange = kk .< rlen                                   # (TILE_M, TILE_K)
        ptrs = (p0 .- rlen) .+ kk
        smask = (kk .>= rlen) .& (kk .< rlen .+ slen)
        sids = ct.gather(inids, ptrs; mask=smask, padding_value=Int32(0))
        ids = ifelse.(inrange, los .+ kk, sids)                # 0 → zero row of B
        w = weights(ids, inrange, ptrs, smask)
        acc = acc .+ gather_b_wsum(B, ids, w, ncols3, BT, TILE_M, TILE_K, TILE_N)
    end
    return acc
end

function spmm_rp_pm_kernel(C::ct.TileArray{T, 2}, rptr::ct.TileArray{Int32, 1},
                           inptr::ct.TileArray{Int32, 1}, inids::ct.TileArray{Int32, 1},
                           B::ct.TileArray{T, 2}, alpha::T, beta::T, rsign::T,
                           TILE_M::Int, TILE_N::Int, TILE_K::Int,
                           BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    acc = rp_accumulate(rptr, inptr, inids, B, bm, bn, BT, TILE_M, TILE_N, TILE_K, T) do ids, inrange, ptrs, smask
        ifelse.(inrange, rsign, -rsign)
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

function spmm_rp_kernel(C::ct.TileArray{T, 2}, rptr::ct.TileArray{Int32, 1},
                        rvals::ct.TileArray{T, 1},
                        inptr::ct.TileArray{Int32, 1}, inids::ct.TileArray{Int32, 1},
                        invals::ct.TileArray{T, 1},
                        B::ct.TileArray{T, 2}, alpha::T, beta::T,
                        TILE_M::Int, TILE_N::Int, TILE_K::Int,
                        BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    acc = rp_accumulate(rptr, inptr, inids, B, bm, bn, BT, TILE_M, TILE_N, TILE_K, T) do ids, inrange, ptrs, smask
        ct.gather(rvals, ids; mask=inrange) .+ ct.gather(invals, ptrs; mask=smask)
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

"build_spmm_rc with the single rptr array; launcher `spmm!(C, rptr, inptr,
inids, rsign, B, α, β)` for `pm = true`, with rvals/invals interposed otherwise."
function build_spmm_rp(::Type{T}; tile_m::Int, tile_n::Int, tile_k::Int, pm::Bool,
                       beta_nz::Bool, bt::Bool=false, num_warps::Int=4) where {T}
    consts = (ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
              ct.Constant{Int, tile_k}, ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    i1 = spmm_ta1(Int32)
    if pm
        k = TritonRun.triton_kernel(spmm_rp_pm_kernel,
            Tuple{spmm_ta2(T), i1, i1, i1, spmm_ta2(T), T, T, T, consts...};
            name="spmm_rp_pm", num_warps)
        return (C, rptr, inptr, inids, rsign, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, rptr, inptr, inids, B, T(α), T(β), T(rsign))
    else
        k = TritonRun.triton_kernel(spmm_rp_kernel,
            Tuple{spmm_ta2(T), i1, spmm_ta1(T), i1, i1, spmm_ta1(T), spmm_ta2(T),
                  T, T, consts...};
            name="spmm_rp", num_warps)
        return (C, rptr, rvals, inptr, inids, invals, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, rptr, rvals, inptr, inids, invals, B, T(α), T(β))
    end
end

# --- harness ----------------------------------------------------------------

const BUILD_CACHE = Dict{Any, Any}()
build_cached(key, build) = get!(() -> try build() catch err; err end,
                                BUILD_CACHE, key)

"Sweep a cuTile kernel over `cands` (β=0 only): each config compiled once
across variants/n (via `key(cfg)`), verified, timed, and printed."
function bench_cutile0(case, label, flops, Cd, Cref, rtol; cands, key, build, launch)
    T = eltype(Cd)
    for cfg in cands
        full = "$label[$(join(cfg, "×"))]"
        spmm! = build_cached(key(cfg), () -> build(cfg))
        if spmm! isa Exception
            fail_row(case, full, spmm!)
            continue
        end
        fill!(Cd, T(NaN))
        launch(spmm!, Cd)
        device_sync()
        check(full, Cd, Cref; rtol) || continue
        t = timeit(() -> launch(spmm!, Cd))
        result_row(case, full, t, flops)
    end
    flush(stdout)
end

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = get(ENV, "SPMM_TYPE", "Float32") == "Float64" ? Float64 : Float32
    datadir = joinpath(@__DIR__, "data")
    data = get(ENV, "SPMM_DATA", "flow_TX")

    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$data.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$(data)_t.jls")))
    m, k = size(A)
    println("# spmm_rptr bench: $BACKEND ($(device_name())), $T, $data: ",
            "A = $m×$k with nnz=$(nnz(A)), n = $ns")

    rc = range_csr(At.colptr, At.rowval, At.nzval)
    no = node_order_rptr(rc, k)
    println("# rsign $(rc.rsign), $(sum(rc.lo .== rc.hi)) empty ranges, ",
            "ranges tile 1:$k in node order")
    rsign = T(rc.rsign)

    lo_j = GPUArr(rc.lo); hi_j = GPUArr(rc.hi)
    inptr_j = GPUArr(rc.inptr); inids_j = GPUArr(rc.inids)
    lo_n = GPUArr(no.lo); hi_n = GPUArr(no.hi); rptr_n = GPUArr(no.rptr)
    inptr_n = GPUArr(no.inptr); inids_n = GPUArr(no.inids)
    # values are ±rsign everywhere, so both row orders share the same arrays
    rvals = GPUArr(fill(rsign, k)); invals = GPUArr(fill(-rsign, length(rc.inids)))

    rng = MersenneTwister(42)
    rtol = sqrt(eps(T)) * 100
    for n in ns
        case = "rptr $data n=$n $T"
        flops = 2.0 * nnz(A) * n
        Bh = rand(rng, T, k, n)
        Crefh = A * Bh
        Crefh_n = Crefh[no.perm, :]
        for (bt, lay) in ((false, ""), (true, " t"))
            Bx = GPUArr(bt ? permutedims(Bh) : Bh)
            Cref_j = GPUArr(bt ? permutedims(Crefh) : Crefh)
            Cref_n = GPUArr(bt ? permutedims(Crefh_n) : Crefh_n)
            Cd = similar(Cref_j)
            backend = KA.get_backend(Cd)
            rc_pm! = spmm_rc_pm_ka!(backend); rp_pm! = spmm_rp_pm_ka!(backend)
            rc_v! = spmm_rc_ka!(backend); rp_v! = spmm_rp_ka!(backend)
            for nb in (1, 4, 8)
                n % nb == 0 || continue
                grid = ka_ndrange(m, n ÷ nb, bt)
                vT0F = (T(1), T(0), Val(nb), Val(bt), Val(false))
                variants = (
                    ("KA rc pm[nb=$nb$lay] lo/hi jds", Cref_j,
                     C -> rc_pm!(C, lo_j, hi_j, inptr_j, inids_j, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)),
                    ("KA rc pm[nb=$nb$lay] lo/hi node", Cref_n,
                     C -> rc_pm!(C, lo_n, hi_n, inptr_n, inids_n, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)),
                    ("KA rc pm[nb=$nb$lay] rptr node", Cref_n,
                     C -> rp_pm!(C, rptr_n, inptr_n, inids_n, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)),
                    ("KA rc[nb=$nb$lay] lo/hi jds", Cref_j,
                     C -> rc_v!(C, lo_j, hi_j, rvals, inptr_j, inids_j, invals, Bx,
                                vT0F...; ndrange=grid)),
                    ("KA rc[nb=$nb$lay] lo/hi node", Cref_n,
                     C -> rc_v!(C, lo_n, hi_n, rvals, inptr_n, inids_n, invals, Bx,
                                vT0F...; ndrange=grid)),
                    ("KA rc[nb=$nb$lay] rptr node", Cref_n,
                     C -> rp_v!(C, rptr_n, rvals, inptr_n, inids_n, invals, Bx,
                                vT0F...; ndrange=grid)))
                for (label, Cref, f0) in variants
                    fill!(Cd, T(NaN))
                    f0(Cd)
                    device_sync()
                    check(label, Cd, Cref; rtol) || continue
                    t = timeit(() -> f0(Cd))
                    result_row(case, label, t, flops)
                end
                flush(stdout)
            end

            # cuTile, same three variants (lo/hi kernels shared across orders)
            cands = rc_tile_candidates(n)
            build_rc = pm -> ((tm, tn, tk, nw),) -> build_spmm_rc(T; tile_m=tm,
                tile_n=tn, tile_k=tk, pm, beta_nz=false, bt, num_warps=nw)
            build_rp = pm -> ((tm, tn, tk, nw),) -> build_spmm_rp(T; tile_m=tm,
                tile_n=tn, tile_k=tk, pm, beta_nz=false, bt, num_warps=nw)
            for (label, Cref, key0, build, launch) in (
                ("cuTile rc pm$lay lo/hi jds", Cref_j, "rc_pm", build_rc(true),
                 (f!, C) -> f!(C, lo_j, hi_j, inptr_j, inids_j, rsign, Bx, 1, 0)),
                ("cuTile rc pm$lay lo/hi node", Cref_n, "rc_pm", build_rc(true),
                 (f!, C) -> f!(C, lo_n, hi_n, inptr_n, inids_n, rsign, Bx, 1, 0)),
                ("cuTile rc pm$lay rptr node", Cref_n, "rp_pm", build_rp(true),
                 (f!, C) -> f!(C, rptr_n, inptr_n, inids_n, rsign, Bx, 1, 0)),
                ("cuTile rc$lay lo/hi jds", Cref_j, "rc_v", build_rc(false),
                 (f!, C) -> f!(C, lo_j, hi_j, rvals, inptr_j, inids_j, invals, Bx, 1, 0)),
                ("cuTile rc$lay lo/hi node", Cref_n, "rc_v", build_rc(false),
                 (f!, C) -> f!(C, lo_n, hi_n, rvals, inptr_n, inids_n, invals, Bx, 1, 0)),
                ("cuTile rc$lay rptr node", Cref_n, "rp_v", build_rp(false),
                 (f!, C) -> f!(C, rptr_n, rvals, inptr_n, inids_n, invals, Bx, 1, 0)))
                bench_cutile0(case, label, flops, Cd, Cref, rtol;
                              cands, key=cfg -> (key0, bt, cfg), build, launch)
            end

            BACKEND == "cuda" && foreach(CUDA.unsafe_free!, (Bx, Cd, Cref_j, Cref_n))
            Bx = Cd = Cref_j = Cref_n = nothing
            GC.gc()
            BACKEND == "cuda" && CUDA.reclaim()
        end
    end
    println("SPMM RPTR BENCH DONE ($BACKEND)")
end

main()
