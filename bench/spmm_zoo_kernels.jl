# cuTile/TileTriton ports of MinimumCostFlows' matrix_zoo SpMM kernels
# (src/matrix_zoo.jl): JDSMatrix / JDSMatrixPM (jagged diagonal storage) and
# Matrix2PerRow / Matrix2PerRowPM (≤2 nonzeros per row at fixed slots).
# Computes C = α·A·B + β·C with B, C dense column-major, like
# spmm_csr_kernels.jl. The PM variants store no values: the sign of the
# column index encodes ±1 (JDS) or slot 1/2 encodes +1/-1 (2-per-row).
#
# Matrix2PerRow{PM}: colidx (and vals) are 2×m matrices passed here as flat
# 1D vectors (vec(colidx)); row i's slots sit at 2i-1 and 2i. A column id of
# 0 means "absent" — the B gather bounds-masks index 0 to a zero tile, so no
# explicit mask is needed. One program covers TILE_M rows × TILE_N columns
# with no inner loop.
#
# JDS{PM}: the j-th nonzero of jagged row i lives at colidx[i + iterptr[j] - 1]
# while that is < iterptr[j+1]. Rows are sorted by decreasing length, so the
# per-tile trip count is the length of the tile's first row, found by a
# scalar while over the diagonal lengths. Within a diagonal the pointers of
# consecutive rows are contiguous, giving coalesced gathers. The jagged→output
# row map is assumed identity (`row::Base.OneTo`), which
# construct_constraint_matrix always produces. Pad iterptr with one trailing
# copy of its last entry so a non-short-circuiting condition evaluation can
# never read past the end.

import cuTile as ct
using TileTriton: TritonRun

const ZOO_SPEC1 = ct.ArraySpec{1}(128, true, (1,), (0,))
const ZOO_SPEC2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
zoo_ta1(T) = ct.TileArray{T, 1, Int32, ZOO_SPEC1}
zoo_ta2(T) = ct.TileArray{T, 2, Int32, ZOO_SPEC2}

# --- Matrix2PerRowPM: values are +1 (slot 1) / -1 (slot 2) ------------------

function spmm_2pr_pm_kernel(C::ct.TileArray{T, 2},
                            colidx::ct.TileArray{Int32, 1},
                            B::ct.TileArray{T, 2},
                            alpha::T, beta::T,
                            TILE_M::Int, TILE_N::Int, BETA_NZ::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = (bm - Int32(1)) * Int32(TILE_M) .+ ct.arange(TILE_M)
    ncols = reshape((bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N),
                    (1, TILE_N))
    j1 = ct.gather(colidx, Int32(2) .* rows .- Int32(1))   # rows past m pad 0
    j2 = ct.gather(colidx, Int32(2) .* rows)
    # column id 0 (absent) bounds-masks to a zero row of B
    b1 = ct.gather(B, (reshape(j1, (TILE_M, 1)), ncols))
    b2 = ct.gather(B, (reshape(j2, (TILE_M, 1)), ncols))
    out = alpha .* (b1 .- b2)
    cidx = (reshape(rows, (TILE_M, 1)), ncols)
    if BETA_NZ
        out = out .+ beta .* ct.gather(C, cidx)
    end
    ct.scatter(C, cidx, out)
    return
end

# --- Matrix2PerRow: explicit values, flat like colidx -----------------------

function spmm_2pr_kernel(C::ct.TileArray{T, 2},
                         colidx::ct.TileArray{Int32, 1},
                         vals::ct.TileArray{T, 1},
                         B::ct.TileArray{T, 2},
                         alpha::T, beta::T,
                         TILE_M::Int, TILE_N::Int, BETA_NZ::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = (bm - Int32(1)) * Int32(TILE_M) .+ ct.arange(TILE_M)
    ncols = reshape((bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N),
                    (1, TILE_N))
    i1 = Int32(2) .* rows .- Int32(1)
    i2 = Int32(2) .* rows
    j1 = ct.gather(colidx, i1)
    j2 = ct.gather(colidx, i2)
    v1 = ct.gather(vals, i1)
    v2 = ct.gather(vals, i2)
    b1 = ct.gather(B, (reshape(j1, (TILE_M, 1)), ncols))
    b2 = ct.gather(B, (reshape(j2, (TILE_M, 1)), ncols))
    out = alpha .* (reshape(v1, (TILE_M, 1)) .* b1 .+
                    reshape(v2, (TILE_M, 1)) .* b2)
    cidx = (reshape(rows, (TILE_M, 1)), ncols)
    if BETA_NZ
        out = out .+ beta .* ct.gather(C, cidx)
    end
    ct.scatter(C, cidx, out)
    return
end

# --- JDSMatrixPM: signed column ids ----------------------------------------

function spmm_jds_pm_kernel(C::ct.TileArray{T, 2},
                            colidx::ct.TileArray{Int32, 1},
                            iterptr::ct.TileArray{Int32, 1},
                            B::ct.TileArray{T, 2},
                            alpha::T, beta::T, ndiag::Int32,
                            TILE_M::Int, TILE_N::Int, BETA_NZ::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    i0 = (bm - Int32(1)) * Int32(TILE_M) + Int32(1)
    rows = (i0 - Int32(1)) .+ ct.arange(TILE_M)
    ncols = reshape((bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N),
                    (1, TILE_N))
    # rows sorted by decreasing length: the tile runs as many diagonals as
    # its first row has nonzeros, i.e. while diagonal j still reaches row i0.
    # Non-short-circuiting `&` keeps this a plain while for the emitter; the
    # sentinel keeps iterptr[nj + 2] in bounds at nj == ndiag.
    nj = Int32(0)
    while (nj < ndiag) & (i0 <= iterptr[nj + Int32(2)] - iterptr[nj + Int32(1)])
        nj += Int32(1)
    end
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nj
        p0 = iterptr[t]
        p1 = iterptr[t + Int32(1)]
        ptrs = (p0 - Int32(1)) .+ rows
        kmask = ptrs .< p1
        k = ct.gather(colidx, ptrs; mask=kmask, padding_value=Int32(0))
        cols = ifelse.(k .< Int32(0), Int32(0) .- k, k)   # id 0 pads B to 0
        sgn = ifelse.(k .< Int32(0), T(-1), T(1))
        bt = ct.gather(B, (reshape(cols, (TILE_M, 1)), ncols))
        acc = acc .+ reshape(sgn, (TILE_M, 1)) .* bt
    end
    cidx = (reshape(rows, (TILE_M, 1)), ncols)
    out = alpha .* acc
    if BETA_NZ
        out = out .+ beta .* ct.gather(C, cidx)
    end
    ct.scatter(C, cidx, out)
    return
end

# --- JDSMatrix: positive column ids + values --------------------------------

function spmm_jds_kernel(C::ct.TileArray{T, 2},
                         colidx::ct.TileArray{Int32, 1},
                         iterptr::ct.TileArray{Int32, 1},
                         nzval::ct.TileArray{T, 1},
                         B::ct.TileArray{T, 2},
                         alpha::T, beta::T, ndiag::Int32,
                         TILE_M::Int, TILE_N::Int, BETA_NZ::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    i0 = (bm - Int32(1)) * Int32(TILE_M) + Int32(1)
    rows = (i0 - Int32(1)) .+ ct.arange(TILE_M)
    ncols = reshape((bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N),
                    (1, TILE_N))
    nj = Int32(0)
    while (nj < ndiag) & (i0 <= iterptr[nj + Int32(2)] - iterptr[nj + Int32(1)])
        nj += Int32(1)
    end
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nj
        p0 = iterptr[t]
        p1 = iterptr[t + Int32(1)]
        ptrs = (p0 - Int32(1)) .+ rows
        kmask = ptrs .< p1
        cols = ct.gather(colidx, ptrs; mask=kmask, padding_value=Int32(0))
        vals = ct.gather(nzval, ptrs; mask=kmask)
        bt = ct.gather(B, (reshape(cols, (TILE_M, 1)), ncols))
        acc = acc .+ reshape(vals, (TILE_M, 1)) .* bt
    end
    cidx = (reshape(rows, (TILE_M, 1)), ncols)
    out = alpha .* acc
    if BETA_NZ
        out = out .+ beta .* ct.gather(C, cidx)
    end
    ct.scatter(C, cidx, out)
    return
end

"""
    build_spmm_2pr(T; tile_m, tile_n, pm, beta_nz) -> spmm!

Compile a Matrix2PerRow{PM} SpMM kernel. The launcher is
`spmm!(C, colidx, B, α, β)` for `pm = true` and
`spmm!(C, colidx, vals, B, α, β)` otherwise, with `colidx`/`vals` the flat
(`vec`) forms of the 2×m matrices.
"""
function build_spmm_2pr(::Type{T}; tile_m::Int, tile_n::Int, pm::Bool,
                        beta_nz::Bool) where {T}
    if pm
        k = TritonRun.triton_kernel(spmm_2pr_pm_kernel,
            Tuple{zoo_ta2(T), zoo_ta1(Int32), zoo_ta2(T), T, T,
                  ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
                  ct.Constant{Bool, beta_nz}};
            name="spmm_2pr_pm", num_warps=4)
        return (C, colidx, B, α, β) ->
            TritonRun.launch!(k, (cld(size(C, 1), tile_m), cld(size(C, 2), tile_n)),
                              C, colidx, B, T(α), T(β))
    else
        k = TritonRun.triton_kernel(spmm_2pr_kernel,
            Tuple{zoo_ta2(T), zoo_ta1(Int32), zoo_ta1(T), zoo_ta2(T), T, T,
                  ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
                  ct.Constant{Bool, beta_nz}};
            name="spmm_2pr", num_warps=4)
        return (C, colidx, vals, B, α, β) ->
            TritonRun.launch!(k, (cld(size(C, 1), tile_m), cld(size(C, 2), tile_n)),
                              C, colidx, vals, B, T(α), T(β))
    end
end

"""
    build_spmm_jds(T; tile_m, tile_n, pm, beta_nz) -> spmm!

Compile a JDSMatrix{PM} SpMM kernel. The launcher is
`spmm!(C, colidx, iterptr, B, α, β)` for `pm = true` and
`spmm!(C, colidx, iterptr, nzval, B, α, β)` otherwise. `iterptr` must carry
one trailing sentinel copy of its last entry (see header). Assumes the
jagged→output row map is the identity.
"""
function build_spmm_jds(::Type{T}; tile_m::Int, tile_n::Int, pm::Bool,
                        beta_nz::Bool) where {T}
    if pm
        k = TritonRun.triton_kernel(spmm_jds_pm_kernel,
            Tuple{zoo_ta2(T), zoo_ta1(Int32), zoo_ta1(Int32), zoo_ta2(T), T, T,
                  Int32, ct.Constant{Int, tile_m}, ct.Constant{Int, tile_n},
                  ct.Constant{Bool, beta_nz}};
            name="spmm_jds_pm", num_warps=4)
        return (C, colidx, iterptr, B, α, β) ->
            TritonRun.launch!(k, (cld(size(C, 1), tile_m), cld(size(C, 2), tile_n)),
                              C, colidx, iterptr, B, T(α), T(β),
                              Int32(length(iterptr) - 2))
    else
        k = TritonRun.triton_kernel(spmm_jds_kernel,
            Tuple{zoo_ta2(T), zoo_ta1(Int32), zoo_ta1(Int32), zoo_ta1(T),
                  zoo_ta2(T), T, T, Int32, ct.Constant{Int, tile_m},
                  ct.Constant{Int, tile_n}, ct.Constant{Bool, beta_nz}};
            name="spmm_jds", num_warps=4)
        return (C, colidx, iterptr, nzval, B, α, β) ->
            TritonRun.launch!(k, (cld(size(C, 1), tile_m), cld(size(C, 2), tile_n)),
                              C, colidx, iterptr, nzval, B, T(α), T(β),
                              Int32(length(iterptr) - 2))
    end
end
