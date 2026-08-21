# cuTile/TileTriton ports of MinimumCostFlows' matrix_zoo SpMM kernels
# (src/matrix_zoo.jl): JDSMatrix / JDSMatrixPM (jagged diagonal storage) and
# Matrix2PerRow / Matrix2PerRowPM (≤2 nonzeros per row at fixed slots).
# Computes C = α·A·B + β·C with B, C dense column-major, like
# spmm_csr_kernels.jl. The PM variants store no values: the sign of the
# column index encodes ±1 (JDS) or slot 1/2 encodes +1/-1 (2-per-row).
#
# Matrix2PerRow{PM}: colidx (and vals) are the natural 2×m matrices; a
# program loads each slot row of its block as a (1, TILE_M) tile via
# `eachtile`. A column id of 0 means "absent" — the B gather bounds-masks
# index 0 to a zero tile, so no explicit mask is needed. One program covers
# TILE_M rows × TILE_N columns with no inner loop.
#
# JDS{PM}: the j-th nonzero of jagged row i lives at colidx[i + iterptr[j] - 1]
# while that is < iterptr[j+1]. Rows are sorted by decreasing length, so the
# per-tile trip count is the length of the tile's first row, found by a
# scalar while over the diagonal lengths. Within a diagonal the pointers of
# consecutive rows are contiguous, giving coalesced gathers. The jagged→output
# row map is assumed identity (`row::Base.OneTo`), which
# construct_constraint_matrix always produces.
#
# Every kernel takes a BT constant selecting the fully transposed layout:
# B passed as n×k and C as n×m, so the rhs columns are contiguous.

import cuTile as ct
using TileTriton: TritonRun

include(joinpath(@__DIR__, "spmm_specs.jl"))

"Global indices covered by block `b` along one axis, as a (TILE,) tile."
@inline tile_span(b, TILE::Int) = (b - Int32(1)) * Int32(TILE) .+ ct.arange(TILE)

"Gather the (TILE_M, TILE_N) B block for the (TILE_M, 1)-shaped column-id
tile `cols`; `BT` picks the transposed layout (B is n×k, the rhs columns
contiguous). Ids ≤ 0 bounds-mask to zero rows either way."
@inline gather_b(B, cols, ncols, BT::Bool) =
    BT ? ct.gather(B, (ncols, cols)) : ct.gather(B, (cols, ncols))

"""
Number of jagged diagonals reaching row `i0`. Rows are sorted by decreasing
length, so a tile starting at `i0` runs as many diagonals as its first row
has nonzeros: count while diagonal j still reaches `i0`. Non-short-circuiting
`&` keeps this a plain while loop for the emitter; the `min` clamp keeps the
speculatively evaluated load in bounds at `nj == ndiag` (`iterptr` has
`ndiag + 1` entries).
"""
@inline function jds_trip_count(iterptr, i0::Int32, ndiag::Int32)
    nj = Int32(0)
    while (nj < ndiag) &
          (i0 <= iterptr[min(nj + Int32(2), ndiag + Int32(1))] - iterptr[nj + Int32(1)])
        nj += Int32(1)
    end
    return nj
end

"α-scale, optional β·C accumulate, and the edge-clipped store of block
(bm, bn) of C, shared by all four kernels. With `BT` the array holds the
transposed n×m C; a permuted view keeps the block logic identical."
@inline function spmm_epilogue(C, bm, bn, acc, alpha, beta,
                               TILE_M::Int, TILE_N::Int, BETA_NZ::Bool, BT::Bool)
    if BT
        epilogue_store(permutedims(C, (2, 1)), bm, bn, acc, alpha, beta,
                       TILE_M, TILE_N, BETA_NZ)
    else
        epilogue_store(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ)
    end
    return
end

@inline function epilogue_store(C, bm, bn, acc, alpha, beta,
                                TILE_M::Int, TILE_N::Int, BETA_NZ::Bool)
    tiles = ct.eachtile(C, (TILE_M, TILE_N); padding_mode=ct.PaddingMode.Zero)
    out = alpha .* acc
    if BETA_NZ
        out = out .+ beta .* ct.load(tiles, (bm, bn))
    end
    ct.store(tiles, (bm, bn), out)
    return
end

# --- Matrix2PerRowPM: values are +1 (slot 1) / -1 (slot 2) ------------------

function spmm_2pr_pm_kernel(C::ct.TileArray{T, 2},
                            colidx::ct.TileArray{Int32, 2},
                            B::ct.TileArray{T, 2},
                            alpha::T, beta::T, TILE_M::Int, TILE_N::Int,
                            BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    slots = ct.eachtile(colidx, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    j1 = ct.load(slots, (Int32(1), bm))      # rows past m pad 0
    j2 = ct.load(slots, (Int32(2), bm))
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    # column id 0 (absent) bounds-masks to a zero row of B
    b1 = gather_b(B, reshape(j1, (TILE_M, 1)), ncols, BT)
    b2 = gather_b(B, reshape(j2, (TILE_M, 1)), ncols, BT)
    spmm_epilogue(C, bm, bn, b1 .- b2, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

# --- Matrix2PerRow: explicit values, 2×m like colidx ------------------------

function spmm_2pr_kernel(C::ct.TileArray{T, 2},
                         colidx::ct.TileArray{Int32, 2},
                         vals::ct.TileArray{T, 2},
                         B::ct.TileArray{T, 2},
                         alpha::T, beta::T, TILE_M::Int, TILE_N::Int,
                         BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    slots = ct.eachtile(colidx, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    vslots = ct.eachtile(vals, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    j1 = ct.load(slots, (Int32(1), bm))
    j2 = ct.load(slots, (Int32(2), bm))
    v1 = ct.load(vslots, (Int32(1), bm))
    v2 = ct.load(vslots, (Int32(2), bm))
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    b1 = gather_b(B, reshape(j1, (TILE_M, 1)), ncols, BT)
    b2 = gather_b(B, reshape(j2, (TILE_M, 1)), ncols, BT)
    acc = reshape(v1, (TILE_M, 1)) .* b1 .+ reshape(v2, (TILE_M, 1)) .* b2
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

# --- JDSMatrixPM: signed column ids ----------------------------------------

function spmm_jds_pm_kernel(C::ct.TileArray{T, 2},
                            colidx::ct.TileArray{Int32, 1},
                            iterptr::ct.TileArray{Int32, 1},
                            B::ct.TileArray{T, 2},
                            alpha::T, beta::T, ndiag::Int32, TILE_M::Int,
                            TILE_N::Int, BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    i0 = (bm - Int32(1)) * Int32(TILE_M) + Int32(1)
    nj = jds_trip_count(iterptr, i0, ndiag)
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nj
        p0 = iterptr[t]
        p1 = iterptr[t + Int32(1)]
        ptrs = (p0 - Int32(1)) .+ rows
        kmask = ptrs .< p1
        k = ct.gather(colidx, ptrs; mask=kmask, padding_value=Int32(0))
        cols = abs.(k)                            # id 0 pads B to 0
        sgn = ifelse.(k .< 0, T(-1), T(1))
        btile = gather_b(B, reshape(cols, (TILE_M, 1)), ncols, BT)
        acc = acc .+ reshape(sgn, (TILE_M, 1)) .* btile
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

# --- JDSMatrix: positive column ids + values --------------------------------

function spmm_jds_kernel(C::ct.TileArray{T, 2},
                         colidx::ct.TileArray{Int32, 1},
                         iterptr::ct.TileArray{Int32, 1},
                         nzval::ct.TileArray{T, 1},
                         B::ct.TileArray{T, 2},
                         alpha::T, beta::T, ndiag::Int32, TILE_M::Int,
                         TILE_N::Int, BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    i0 = (bm - Int32(1)) * Int32(TILE_M) + Int32(1)
    nj = jds_trip_count(iterptr, i0, ndiag)
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nj
        p0 = iterptr[t]
        p1 = iterptr[t + Int32(1)]
        ptrs = (p0 - Int32(1)) .+ rows
        kmask = ptrs .< p1
        cols = ct.gather(colidx, ptrs; mask=kmask, padding_value=Int32(0))
        vals = ct.gather(nzval, ptrs; mask=kmask)
        btile = gather_b(B, reshape(cols, (TILE_M, 1)), ncols, BT)
        acc = acc .+ reshape(vals, (TILE_M, 1)) .* btile
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

"""
    build_spmm_2pr(T; tile_m, tile_n, pm, beta_nz) -> spmm!

Compile a Matrix2PerRow{PM} SpMM kernel (`bt = true` for the transposed
n×k B layout). The launcher is
`spmm!(C, colidx, B, α, β)` for `pm = true` and
`spmm!(C, colidx, vals, B, α, β)` otherwise, with `colidx`/`vals` the
2×m slot matrices.
"""
function build_spmm_2pr(::Type{T}; tile_m::Int, tile_n::Int, pm::Bool,
                        beta_nz::Bool, bt::Bool=false, num_warps::Int=4) where {T}
    consts = (ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
              ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    if pm
        k = TritonRun.triton_kernel(spmm_2pr_pm_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), T, T, consts...};
            name="spmm_2pr_pm", num_warps)
        return (C, colidx, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, B, T(α), T(β))
    else
        k = TritonRun.triton_kernel(spmm_2pr_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), spmm_ta2(T), T, T,
                  consts...};
            name="spmm_2pr", num_warps)
        return (C, colidx, vals, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, vals, B, T(α), T(β))
    end
end

"""
    build_spmm_jds(T; tile_m, tile_n, pm, beta_nz) -> spmm!

Compile a JDSMatrix{PM} SpMM kernel (`bt = true` for the transposed n×k
B layout). The launcher is
`spmm!(C, colidx, iterptr, B, α, β)` for `pm = true` and
`spmm!(C, colidx, iterptr, nzval, B, α, β)` otherwise. Assumes the
jagged→output row map is the identity.
"""
function build_spmm_jds(::Type{T}; tile_m::Int, tile_n::Int, pm::Bool,
                        beta_nz::Bool, bt::Bool=false, num_warps::Int=4) where {T}
    # runtime ndiag, then the compile-time tile constants
    tail = (Int32, ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
            ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    if pm
        k = TritonRun.triton_kernel(spmm_jds_pm_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta2(T),
                  T, T, tail...};
            name="spmm_jds_pm", num_warps)
        return (C, colidx, iterptr, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, iterptr, B, T(α), T(β),
                              Int32(length(iterptr) - 1))
    else
        k = TritonRun.triton_kernel(spmm_jds_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta1(T),
                  spmm_ta2(T), T, T, tail...};
            name="spmm_jds", num_warps)
        return (C, colidx, iterptr, nzval, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, iterptr, nzval, B, T(α), T(β),
                              Int32(length(iterptr) - 1))
    end
end

# --- Range + CSR ("rc"): the node-arc incidence seen from the rows ---------
#
# With arcs numbered by one endpoint (the CSC order of the adjacency
# matrix), each row of the nodes×arcs incidence is one contiguous column
# range (the arcs of that endpoint, one sign) plus a scattered list (the
# arcs of the other endpoint, the opposite sign). The range needs no column
# ids at all — its B rows are a dense strip — so only the scattered half is
# a CSR. `lo`/`hi` bound the range per row (hi exclusive), `rsign` is the
# value of the range entries (the scattered ones are -rsign). The vals
# variant carries `rvals` indexed by column (each column lies in exactly one
# range) and `invals` per scattered entry instead.

"Gather the (TILE_M, TILE_K, TILE_N) B block for the (TILE_M, TILE_K) id
tile and sum it over K; ids ≤ 0 bounds-mask to zero rows (see gather_b)."
@inline function gather_b_sum(B, ids, ncols3, BT::Bool, TILE_M::Int, TILE_K::Int, TILE_N::Int)
    ids3 = reshape(ids, (TILE_M, TILE_K, 1))
    bt = BT ? ct.gather(B, (ncols3, ids3)) : ct.gather(B, (ids3, ncols3))
    return reshape(sum(bt; dims=2), (TILE_M, TILE_N))
end

@inline function gather_b_wsum(B, ids, w, ncols3, BT::Bool, TILE_M::Int, TILE_K::Int, TILE_N::Int)
    ids3 = reshape(ids, (TILE_M, TILE_K, 1))
    bt = BT ? ct.gather(B, (ncols3, ids3)) : ct.gather(B, (ids3, ncols3))
    return reshape(sum(reshape(w, (TILE_M, TILE_K, 1)) .* bt; dims=2), (TILE_M, TILE_N))
end

function spmm_rc_pm_kernel(C::ct.TileArray{T, 2},
                           lo::ct.TileArray{Int32, 1}, hi::ct.TileArray{Int32, 1},
                           inptr::ct.TileArray{Int32, 1}, inids::ct.TileArray{Int32, 1},
                           B::ct.TileArray{T, 2}, alpha::T, beta::T, rsign::T,
                           TILE_M::Int, TILE_N::Int, TILE_K::Int,
                           BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    ncols3 = reshape(tile_span(bn, TILE_N), (1, 1, TILE_N))
    ks = reshape(ct.arange(TILE_K) .- Int32(1), (1, TILE_K))
    # contiguous range: the ids are the range itself (rows past m pad 0 → empty)
    los = reshape(ct.gather(lo, rows), (TILE_M, 1))
    his = reshape(ct.gather(hi, rows), (TILE_M, 1))
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):cld(maximum(his - los), Int32(TILE_K))
        ids = (los .+ (t - Int32(1)) * Int32(TILE_K)) .+ ks
        ids = ifelse.(ids .< his, ids, Int32(0))
        acc = acc .+ gather_b_sum(B, ids, ncols3, BT, TILE_M, TILE_K, TILE_N)
    end
    # scattered columns: a CSR
    p0 = reshape(ct.gather(inptr, rows), (TILE_M, 1))
    p1 = reshape(ct.gather(inptr, rows .+ Int32(1)), (TILE_M, 1))
    acc2 = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):cld(maximum(p1 - p0), Int32(TILE_K))
        ptrs = (p0 .+ (t - Int32(1)) * Int32(TILE_K)) .+ ks
        ids = ct.gather(inids, ptrs; mask=ptrs .< p1, padding_value=Int32(0))
        acc2 = acc2 .+ gather_b_sum(B, ids, ncols3, BT, TILE_M, TILE_K, TILE_N)
    end
    spmm_epilogue(C, bm, bn, rsign .* (acc .- acc2), alpha, beta,
                  TILE_M, TILE_N, BETA_NZ, BT)
    return
end

function spmm_rc_kernel(C::ct.TileArray{T, 2},
                        lo::ct.TileArray{Int32, 1}, hi::ct.TileArray{Int32, 1},
                        rvals::ct.TileArray{T, 1},
                        inptr::ct.TileArray{Int32, 1}, inids::ct.TileArray{Int32, 1},
                        invals::ct.TileArray{T, 1},
                        B::ct.TileArray{T, 2}, alpha::T, beta::T,
                        TILE_M::Int, TILE_N::Int, TILE_K::Int,
                        BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    ncols3 = reshape(tile_span(bn, TILE_N), (1, 1, TILE_N))
    ks = reshape(ct.arange(TILE_K) .- Int32(1), (1, TILE_K))
    los = reshape(ct.gather(lo, rows), (TILE_M, 1))
    his = reshape(ct.gather(hi, rows), (TILE_M, 1))
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):cld(maximum(his - los), Int32(TILE_K))
        ids = (los .+ (t - Int32(1)) * Int32(TILE_K)) .+ ks
        ids = ifelse.(ids .< his, ids, Int32(0))
        w = ct.gather(rvals, ids)                 # id 0 bounds-masks to 0
        acc = acc .+ gather_b_wsum(B, ids, w, ncols3, BT, TILE_M, TILE_K, TILE_N)
    end
    p0 = reshape(ct.gather(inptr, rows), (TILE_M, 1))
    p1 = reshape(ct.gather(inptr, rows .+ Int32(1)), (TILE_M, 1))
    for t in Int32(1):cld(maximum(p1 - p0), Int32(TILE_K))
        ptrs = (p0 .+ (t - Int32(1)) * Int32(TILE_K)) .+ ks
        kmask = ptrs .< p1
        ids = ct.gather(inids, ptrs; mask=kmask, padding_value=Int32(0))
        w = ct.gather(invals, ptrs; mask=kmask)
        acc = acc .+ gather_b_wsum(B, ids, w, ncols3, BT, TILE_M, TILE_K, TILE_N)
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

"""
    build_spmm_rc(T; tile_m, tile_n, tile_k, pm, beta_nz, bt, num_warps) -> spmm!

Compile a range + CSR SpMM kernel. The launcher is
`spmm!(C, lo, hi, inptr, inids, rsign, B, α, β)` for `pm = true` and
`spmm!(C, lo, hi, rvals, inptr, inids, invals, B, α, β)` otherwise.
"""
function build_spmm_rc(::Type{T}; tile_m::Int, tile_n::Int, tile_k::Int, pm::Bool,
                       beta_nz::Bool, bt::Bool=false, num_warps::Int=4) where {T}
    consts = (ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
              ct.Constant{Int, tile_k}, ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    i1 = spmm_ta1(Int32)
    if pm
        k = TritonRun.triton_kernel(spmm_rc_pm_kernel,
            Tuple{spmm_ta2(T), i1, i1, i1, i1, spmm_ta2(T), T, T, T, consts...};
            name="spmm_rc_pm", num_warps)
        return (C, lo, hi, inptr, inids, rsign, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, lo, hi, inptr, inids, B, T(α), T(β), T(rsign))
    else
        k = TritonRun.triton_kernel(spmm_rc_kernel,
            Tuple{spmm_ta2(T), i1, i1, spmm_ta1(T), i1, i1, spmm_ta1(T), spmm_ta2(T),
                  T, T, consts...};
            name="spmm_rc", num_warps)
        return (C, lo, hi, rvals, inptr, inids, invals, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, lo, hi, rvals, inptr, inids, invals, B, T(α), T(β))
    end
end

# rc candidates: (tile_m, tile_n, tile_k, num_warps); each iteration gathers
# tile_k B rows per C row, so tile_k plays per_row's role in the cap.
rc_tile_candidates(n; tile_k=4) =
    [(tm, tn, tile_k, nw) for (tm, tn, nw) in zoo_tile_candidates(n; per_row=tile_k)]

# Candidate (tile_m, tile_n, num_warps) configs. The register footprint is
# capped at 4096 gathered B elements per program at 4 warps (`per_row` B
# rows gathered per C row), scaled with the warp count. Wide n is offered
# as narrow slabs too (tile_n = 8..32): the gather-latency-bound kernels
# want few elements per thread — 16×32 at 8 warps (2 per thread) is the
# 2pr winner on the flow matrix, and tile_m ≥ 128 never won, so the
# list stops at 64 rows (spmm_flow_results.md, follow-up 3).
function zoo_tile_candidates(n; per_row)
    tn_max = clamp(nextpow(2, n), 4, 64)
    tns = unique(clamp.((8, 16, 32, tn_max), 4, tn_max))
    return [(tm, tn, nw) for nw in (4, 8), tn in tns, tm in (16, 32, 64)
            if tm * tn <= 4096 ÷ per_row * (nw ÷ 4)]
end
