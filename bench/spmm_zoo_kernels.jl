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
@inline gather_b(B, cols, ncols, BT::Bool, check_bounds::Bool=true) =
    BT ? ct.gather(B, (ncols, cols); check_bounds) :
         ct.gather(B, (cols, ncols); check_bounds)

"""
Slot ids for the 2-per-row kernels. With `EXACT2` every row is known to
hold two entries (the incidence matrix: each arc has a tail and a head), so
the ids need no bounds-masking on the B gather; the rows past m that
`eachtile` pads with 0 are clamped to row 1 of B instead (their result is
discarded by the masked store). Without it, id 0 bounds-masks to a zero
row of B.
"""
@inline slot_ids(j, EXACT2::Bool) = EXACT2 ? max.(j, Int32(1)) : j

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
                            BETA_NZ::Bool, BT::Bool, EXACT2::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    slots = ct.eachtile(colidx, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    j1 = slot_ids(ct.load(slots, (Int32(1), bm)), EXACT2)   # rows past m pad 0
    j2 = slot_ids(ct.load(slots, (Int32(2), bm)), EXACT2)
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    # column id 0 (absent) bounds-masks to a zero row of B
    b1 = gather_b(B, reshape(j1, (TILE_M, 1)), ncols, BT, !EXACT2)
    b2 = gather_b(B, reshape(j2, (TILE_M, 1)), ncols, BT, !EXACT2)
    spmm_epilogue(C, bm, bn, b1 .- b2, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
    return
end

# --- Matrix2PerRow: explicit values, 2×m like colidx ------------------------

function spmm_2pr_kernel(C::ct.TileArray{T, 2},
                         colidx::ct.TileArray{Int32, 2},
                         vals::ct.TileArray{T, 2},
                         B::ct.TileArray{T, 2},
                         alpha::T, beta::T, TILE_M::Int, TILE_N::Int,
                         BETA_NZ::Bool, BT::Bool, EXACT2::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    slots = ct.eachtile(colidx, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    vslots = ct.eachtile(vals, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    j1 = slot_ids(ct.load(slots, (Int32(1), bm)), EXACT2)
    j2 = slot_ids(ct.load(slots, (Int32(2), bm)), EXACT2)
    v1 = ct.load(vslots, (Int32(1), bm))
    v2 = ct.load(vslots, (Int32(2), bm))
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    b1 = gather_b(B, reshape(j1, (TILE_M, 1)), ncols, BT, !EXACT2)
    b2 = gather_b(B, reshape(j2, (TILE_M, 1)), ncols, BT, !EXACT2)
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
    build_spmm_2pr(T; tile_m, tile_n, pm, beta_nz, bt, exact2) -> spmm!

Compile a Matrix2PerRow{PM} SpMM kernel (`bt = true` for the transposed
n×k B layout; `exact2 = true` asserts two entries in every row and drops
the B-gather bounds checks, which also needs `tile_n` to divide n). The
launcher is
`spmm!(C, colidx, B, α, β)` for `pm = true` and
`spmm!(C, colidx, vals, B, α, β)` otherwise, with `colidx`/`vals` the
2×m slot matrices.
"""
function build_spmm_2pr(::Type{T}; tile_m::Int, tile_n::Int, pm::Bool,
                        beta_nz::Bool, bt::Bool=false, num_warps::Int=4,
                        exact2::Bool=false) where {T}
    consts = (ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
              ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt},
              ct.Constant{Bool, exact2})
    grid = bt ? spmm_grid_t : spmm_grid
    # unchecked gathers read B[:, col] for every tile column, so n % tile_n == 0
    checkn(B) = !exact2 || size(B, bt ? 1 : 2) % tile_n == 0 ||
        throw(ArgumentError("exact2 needs tile_n = $tile_n to divide n = $(size(B, bt ? 1 : 2))"))
    if pm
        k = TritonRun.triton_kernel(spmm_2pr_pm_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), T, T, consts...};
            name=exact2 ? "spmm_2pr_pm_x2" : "spmm_2pr_pm", num_warps)
        return (C, colidx, B, α, β) ->
            (checkn(B); TritonRun.launch!(k, grid(C, tile_m, tile_n),
                                          C, colidx, B, T(α), T(β)))
    else
        k = TritonRun.triton_kernel(spmm_2pr_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), spmm_ta2(T), T, T,
                  consts...};
            name=exact2 ? "spmm_2pr_x2" : "spmm_2pr", num_warps)
        return (C, colidx, vals, B, α, β) ->
            (checkn(B); TritonRun.launch!(k, grid(C, tile_m, tile_n),
                                          C, colidx, vals, B, T(α), T(β)))
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
# a CSR. The kernels walk the k-th nonzero of every row in one loop (see
# rc_accumulate), so a tile costs as many iterations as its longest row. `lo`/`hi` bound the range per row (hi exclusive), `rsign` is the
# value of the range entries (the scattered ones are -rsign). The vals
# variant carries `rvals` indexed by column (each column lies in exactly one
# range) and `invals` per scattered entry instead.

@inline function gather_b_wsum(B, ids, w, ncols3, BT::Bool, TILE_M::Int, TILE_K::Int, TILE_N::Int)
    ids3 = reshape(ids, (TILE_M, TILE_K, 1))
    bt = BT ? ct.gather(B, (ncols3, ids3)) : ct.gather(B, (ids3, ncols3))
    return reshape(sum(reshape(w, (TILE_M, TILE_K, 1)) .* bt; dims=2), (TILE_M, TILE_N))
end

"""
Range + CSR body shared by the pm and vals kernels: one loop over the
k-th nonzero of each row, k = 0 … total−1. Entry k is `lo + k` while
k < rangelen (no id load) and the CSR entry `inids[p0 + k − rangelen]`
after that. Since A's rows are sorted by total length, a tile's trip count
is its longest row — the same lane utilisation as JDS, with half the id
gathers gone. `body(ids, inrange, ptrs, smask)` returns the weights for
the (TILE_M, TILE_K) block.
"""
@inline function rc_accumulate(weights, lo, hi, inptr, inids, B, bm, bn, BT::Bool,
                               TILE_M::Int, TILE_N::Int, TILE_K::Int, ::Type{T}) where {T}
    rows = tile_span(bm, TILE_M)
    ncols3 = reshape(tile_span(bn, TILE_N), (1, 1, TILE_N))
    ks = reshape(ct.arange(TILE_K) .- Int32(1), (1, TILE_K))
    lo1 = ct.gather(lo, rows)                              # rows past m pad 0 → empty
    rlen1 = ct.gather(hi, rows) .- lo1
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

function spmm_rc_pm_kernel(C::ct.TileArray{T, 2},
                           lo::ct.TileArray{Int32, 1}, hi::ct.TileArray{Int32, 1},
                           inptr::ct.TileArray{Int32, 1}, inids::ct.TileArray{Int32, 1},
                           B::ct.TileArray{T, 2}, alpha::T, beta::T, rsign::T,
                           TILE_M::Int, TILE_N::Int, TILE_K::Int,
                           BETA_NZ::Bool, BT::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    # range entries rsign, scattered ones -rsign (ids of 0 contribute nothing)
    acc = rc_accumulate(lo, hi, inptr, inids, B, bm, bn, BT, TILE_M, TILE_N, TILE_K, T) do ids, inrange, ptrs, smask
        ifelse.(inrange, rsign, -rsign)
    end
    spmm_epilogue(C, bm, bn, acc, alpha, beta, TILE_M, TILE_N, BETA_NZ, BT)
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
    acc = rc_accumulate(lo, hi, inptr, inids, B, bm, bn, BT, TILE_M, TILE_N, TILE_K, T) do ids, inrange, ptrs, smask
        # range weights by column id (masked to the range), scattered by entry
        ct.gather(rvals, ids; mask=inrange) .+ ct.gather(invals, ptrs; mask=smask)
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

# --- heavy rows of a range+CSR matrix: chunk-parallel partial sums ----------
#
# Rows far longer than the rest (the max-flow source/sink nodes of the vision
# instances: ~92k entries vs ≤ 14) serialize a row-per-program kernel on one
# program. The hybrid ("rch") keeps those rows empty in the rc arrays and
# handles them here: pass 1 splits each heavy row's [range; scattered]
# sequence into CHUNK-entry chunks (chunk c of heavy row h = crow[c], local
# index c - cptr[h]) and writes one partial row of n sums per chunk; pass 2
# sums the chunks of each heavy row into C (which the rc kernel has already
# set to α·0 + β·C for those rows).

@inline function heavy_accumulate(weights, crow, cptr, hlo, hhi, hptr, hids, B, c, bn,
                                  BT::Bool, TILE_N::Int, TILE_K::Int, CHUNK::Int,
                                  ::Type{T}) where {T}
    cs = tile_span(c, 1)                                  # (1,) chunk index
    ncols3 = reshape(tile_span(bn, TILE_N), (1, 1, TILE_N))
    ks = reshape(ct.arange(TILE_K) .- Int32(1), (1, TILE_K))
    h1 = ct.gather(crow, cs)
    k01 = (cs .- ct.gather(cptr, h1)) .* Int32(CHUNK)     # first entry of this chunk
    lo1 = ct.gather(hlo, h1); rlen1 = ct.gather(hhi, h1) .- lo1
    p01 = ct.gather(hptr, h1); slen1 = ct.gather(hptr, h1 .+ Int32(1)) .- p01
    k0 = reshape(k01, (1, 1)); los = reshape(lo1, (1, 1)); rlen = reshape(rlen1, (1, 1))
    p0 = reshape(p01, (1, 1)); slen = reshape(slen1, (1, 1))
    acc = zeros(T, (1, TILE_N))
    for t in Int32(1):Int32(CHUNK ÷ TILE_K)
        kk = (k0 .+ (t - Int32(1)) * Int32(TILE_K)) .+ ks
        inrange = kk .< rlen
        ptrs = (p0 .- rlen) .+ kk
        smask = (kk .>= rlen) .& (kk .< rlen .+ slen)
        sids = ct.gather(hids, ptrs; mask=smask, padding_value=Int32(0))
        ids = ifelse.(inrange, los .+ kk, sids)           # 0 past the row: masked B row
        w = weights(ids, inrange, ptrs, smask)
        acc = acc .+ gather_b_wsum(B, ids, w, ncols3, BT, 1, TILE_K, TILE_N)
    end
    return acc
end

function spmm_heavy_pm_kernel(part::ct.TileArray{T, 2}, crow::ct.TileArray{Int32, 1},
                              cptr::ct.TileArray{Int32, 1}, hlo::ct.TileArray{Int32, 1},
                              hhi::ct.TileArray{Int32, 1}, hptr::ct.TileArray{Int32, 1},
                              hids::ct.TileArray{Int32, 1}, B::ct.TileArray{T, 2},
                              rsign::T, TILE_N::Int, TILE_K::Int, CHUNK::Int,
                              BT::Bool) where {T}
    c = ct.bid(1)
    bn = ct.bid(2)
    weights = (ids, inrange, ptrs, smask) -> ifelse.(inrange, rsign, -rsign)
    acc = heavy_accumulate(weights, crow, cptr, hlo, hhi, hptr, hids, B, c, bn, BT,
                           TILE_N, TILE_K, CHUNK, T)
    spmm_epilogue(part, c, bn, acc, one(T), zero(T), 1, TILE_N, false, BT)
    return
end

function spmm_heavy_kernel(part::ct.TileArray{T, 2}, crow::ct.TileArray{Int32, 1},
                           cptr::ct.TileArray{Int32, 1}, hlo::ct.TileArray{Int32, 1},
                           hhi::ct.TileArray{Int32, 1}, rvals::ct.TileArray{T, 1},
                           hptr::ct.TileArray{Int32, 1}, hids::ct.TileArray{Int32, 1},
                           hinvals::ct.TileArray{T, 1}, B::ct.TileArray{T, 2},
                           TILE_N::Int, TILE_K::Int, CHUNK::Int, BT::Bool) where {T}
    c = ct.bid(1)
    bn = ct.bid(2)
    weights = (ids, inrange, ptrs, smask) ->
        ct.gather(rvals, ids; mask=inrange) .+ ct.gather(hinvals, ptrs; mask=smask)
    acc = heavy_accumulate(weights, crow, cptr, hlo, hhi, hptr, hids, B, c, bn, BT,
                           TILE_N, TILE_K, CHUNK, T)
    spmm_epilogue(part, c, bn, acc, one(T), zero(T), 1, TILE_N, false, BT)
    return
end

"Pass 2: C[hrow[h], :] += alpha · Σ chunks of h. One program per (heavy row,
column block), TILE_C chunk partials per step."
function spmm_heavy_reduce_kernel(C::ct.TileArray{T, 2}, part::ct.TileArray{T, 2},
                                  hrow::ct.TileArray{Int32, 1}, cptr::ct.TileArray{Int32, 1},
                                  alpha::T, TILE_C::Int, TILE_N::Int, BT::Bool) where {T}
    h = ct.bid(1)
    bn = ct.bid(2)
    hs = tile_span(h, 1)
    c01 = ct.gather(cptr, hs)
    c11 = ct.gather(cptr, hs .+ Int32(1))
    nt = cld(maximum(c11 .- c01), Int32(TILE_C))
    c0 = reshape(c01, (1, 1)); c1 = reshape(c11, (1, 1))
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    cidx = reshape(ct.arange(TILE_C) .- Int32(1), (TILE_C, 1))
    acc = zeros(T, (TILE_C, TILE_N))
    for t in Int32(1):nt
        cc = (c0 .+ (t - Int32(1)) * Int32(TILE_C)) .+ cidx
        mask = (cc .< c1) .& (ncols .> Int32(0))
        acc = acc .+ (BT ? ct.gather(part, (ncols, cc); mask) : ct.gather(part, (cc, ncols); mask))
    end
    s = sum(acc; dims=1)
    rows = reshape(ct.gather(hrow, hs), (1, 1))
    if BT
        ct.scatter(C, (ncols, rows), ct.gather(C, (ncols, rows)) .+ alpha .* s)
    else
        ct.scatter(C, (rows, ncols), ct.gather(C, (rows, ncols)) .+ alpha .* s)
    end
    return
end

"""
    build_spmm_heavy(T; tile_n, tile_k, chunk, pm, bt) -> (partials!, reduce!)

Heavy-row passes of the hybrid range+CSR format.
`partials!(part, crow, cptr, hlo, hhi, hptr, hids, rsign, B)` (pm) /
`partials!(part, crow, cptr, hlo, hhi, rvals, hptr, hids, hinvals, B)` fills
the (nchunks × n; n × nchunks when `bt`) partial matrix;
`reduce!(C, part, hrow, cptr, α)` adds the heavy rows into C. `chunk`
(a multiple of `tile_k`) is the number of nonzeros per chunk.
"""
function build_spmm_heavy(::Type{T}; tile_n::Int, tile_k::Int, chunk::Int, pm::Bool,
                          bt::Bool=false, num_warps::Int=4, tile_c::Int=64) where {T}
    chunk % tile_k == 0 || throw(ArgumentError("chunk must be a multiple of tile_k"))
    i1 = spmm_ta1(Int32); t1 = spmm_ta1(T); t2 = spmm_ta2(T)
    consts = (ct.Constant{Int, tile_n}, ct.Constant{Int, tile_k}, ct.Constant{Int, chunk},
              ct.Constant{Bool, bt})
    grid(part, nrows) = (Int(nrows), cld(size(part, bt ? 1 : 2), tile_n))
    partials! = if pm
        k = TritonRun.triton_kernel(spmm_heavy_pm_kernel,
            Tuple{t2, i1, i1, i1, i1, i1, i1, t2, T, consts...};
            name="spmm_heavy_pm", num_warps)
        (part, crow, cptr, hlo, hhi, hptr, hids, rsign, B) ->
            TritonRun.launch!(k, grid(part, length(crow)),
                              part, crow, cptr, hlo, hhi, hptr, hids, B, T(rsign))
    else
        k = TritonRun.triton_kernel(spmm_heavy_kernel,
            Tuple{t2, i1, i1, i1, i1, t1, i1, i1, t1, t2, consts...};
            name="spmm_heavy", num_warps)
        (part, crow, cptr, hlo, hhi, rvals, hptr, hids, hinvals, B) ->
            TritonRun.launch!(k, grid(part, length(crow)),
                              part, crow, cptr, hlo, hhi, rvals, hptr, hids, hinvals, B)
    end
    kr = TritonRun.triton_kernel(spmm_heavy_reduce_kernel,
        Tuple{t2, t2, i1, i1, T, ct.Constant{Int, tile_c}, ct.Constant{Int, tile_n},
              ct.Constant{Bool, bt}};
        name="spmm_heavy_reduce", num_warps)
    reduce! = (C, part, hrow, cptr, α) ->
        TritonRun.launch!(kr, grid(part, length(hrow)), C, part, hrow, cptr, T(α))
    return partials!, reduce!
end

# rc candidates: (tile_m, tile_n, tile_k, num_warps); each iteration gathers
# tile_k B rows per C row, so tile_k plays per_row's role in the cap.
rc_tile_candidates(n; tile_ks=(1, 2)) =
    [(tm, tn, tk, nw) for tk in tile_ks for (tm, tn, nw) in zoo_tile_candidates(n; per_row=tk)]

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
