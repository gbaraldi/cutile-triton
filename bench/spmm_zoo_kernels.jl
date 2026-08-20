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
                        beta_nz::Bool, bt::Bool=false) where {T}
    consts = (ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
              ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    if pm
        k = TritonRun.triton_kernel(spmm_2pr_pm_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), T, T, consts...};
            name="spmm_2pr_pm", num_warps=4)
        return (C, colidx, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, B, T(α), T(β))
    else
        k = TritonRun.triton_kernel(spmm_2pr_kernel,
            Tuple{spmm_ta2(T), spmm_ta2(Int32), spmm_ta2(T), spmm_ta2(T), T, T,
                  consts...};
            name="spmm_2pr", num_warps=4)
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
                        beta_nz::Bool, bt::Bool=false) where {T}
    # runtime ndiag, then the compile-time tile constants
    tail = (Int32, ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
            ct.Constant{Bool, beta_nz}, ct.Constant{Bool, bt})
    grid = bt ? spmm_grid_t : spmm_grid
    if pm
        k = TritonRun.triton_kernel(spmm_jds_pm_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta2(T),
                  T, T, tail...};
            name="spmm_jds_pm", num_warps=4)
        return (C, colidx, iterptr, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, iterptr, B, T(α), T(β),
                              Int32(length(iterptr) - 1))
    else
        k = TritonRun.triton_kernel(spmm_jds_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta1(T),
                  spmm_ta2(T), T, T, tail...};
            name="spmm_jds", num_warps=4)
        return (C, colidx, iterptr, nzval, B, α, β) ->
            TritonRun.launch!(k, grid(C, tile_m, tile_n),
                              C, colidx, iterptr, nzval, B, T(α), T(β),
                              Int32(length(iterptr) - 1))
    end
end

# Candidate (tile_m, tile_n) configs, register footprint capped at 4096
# gathered B elements per program (`per_row` B rows gathered per C row).
function zoo_tile_candidates(n; per_row)
    tn = clamp(nextpow(2, n), 4, 64)
    cap = 4096 ÷ per_row
    return [(tm, tn) for tm in (32, 64, 128, 256) if tm * tn <= cap]
end
