# CSR SpMM tile kernels (cuTile), launched through TileTriton on CUDA or ROCm.
#
# Computes C = α·A·B + β·C where A is m×k sparse in CSR with 1-based Int32
# rowptr/colval (the layout of CoolPDLP's GPUSparseMatrixCSR), B dense k×n,
# C dense m×n, both column-major.
#
# Two shapes of parallelism, picked by tile_m:
#  * spmm_csr_row_kernel  (tile_m == 1) — one program per (row of A,
#    TILE_N-wide slab of B columns). The row's nonzeros stream through
#    TILE_K-wide gathers; each nonzero's B row slab is gathered as a
#    (TILE_K, TILE_N) tile and accumulated.
#  * spmm_csr_rows_kernel (tile_m > 1) — TILE_M rows per program, all rows
#    advancing together to the longest row's nonzero count with per-row
#    masks. Better occupancy when rows are short (the PDLP case), at the
#    cost of wasted lanes when row lengths within a block are skewed.
#
# Out-of-range lanes never fault: gathers are bounds-masked, padded column
# ids point at column 1 while their values are padded to 0, and the final
# store/scatter is bounds-checked.

import cuTile as ct
using TileTriton: TritonRun

include(joinpath(@__DIR__, "spmm_specs.jl"))

function spmm_csr_row_kernel(C::ct.TileArray{T, 2},
                             rowptr::ct.TileArray{Int32, 1},
                             colval::ct.TileArray{Int32, 1},
                             nzval::ct.TileArray{T, 1},
                             B::ct.TileArray{T, 2},
                             alpha::T, beta::T,
                             TILE_N::Int, TILE_K::Int, BETA_NZ::Bool) where {T}
    row = ct.bid(1)
    bn = ct.bid(2)
    p0 = rowptr[row]                # first nonzero of the row (1-based)
    p1 = rowptr[row + Int32(1)]     # one past the last
    ncols = (bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N)
    ks = ct.arange(TILE_K) .- Int32(1)
    acc = zeros(T, (TILE_K, TILE_N))
    nk = cld(p1 - p0, Int32(TILE_K))
    for t in Int32(1):nk
        kidx = (p0 + (t - Int32(1)) * Int32(TILE_K)) .+ ks
        kmask = kidx .< p1
        vals = ct.gather(nzval, kidx; mask=kmask)
        cols = ct.gather(colval, kidx; mask=kmask, padding_value=Int32(1))
        bt = ct.gather(B, (reshape(cols, (TILE_K, 1)), reshape(ncols, (1, TILE_N))))
        acc = acc .+ reshape(vals, (TILE_K, 1)) .* bt
    end
    out = alpha .* sum(acc; dims=1)         # (1, TILE_N)
    if BETA_NZ
        cold = ct.load(C; index=(row, bn), shape=(1, TILE_N),
                       padding_mode=ct.PaddingMode.Zero)
        out = out .+ beta .* cold
    end
    ct.store(C; index=(row, bn), tile=out)
    return
end

function spmm_csr_rows_kernel(C::ct.TileArray{T, 2},
                              rowptr::ct.TileArray{Int32, 1},
                              colval::ct.TileArray{Int32, 1},
                              nzval::ct.TileArray{T, 1},
                              B::ct.TileArray{T, 2},
                              alpha::T, beta::T,
                              TILE_M::Int, TILE_N::Int, TILE_K::Int,
                              BETA_NZ::Bool) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = (bm - Int32(1)) * Int32(TILE_M) .+ ct.arange(TILE_M)
    ncols = (bn - Int32(1)) * Int32(TILE_N) .+ ct.arange(TILE_N)
    # rows past m gather rowptr's padding (0), giving them zero nonzeros
    p0s = ct.gather(rowptr, rows)
    p1s = ct.gather(rowptr, rows .+ Int32(1))
    nk = cld(maximum(p1s - p0s), Int32(TILE_K))
    ks = ct.arange(TILE_K) .- Int32(1)
    p0m = reshape(p0s, (TILE_M, 1))
    p1m = reshape(p1s, (TILE_M, 1))
    acc = zeros(T, (TILE_M, TILE_N))
    for t in Int32(1):nk
        kidx = (p0m .+ (t - Int32(1)) * Int32(TILE_K)) .+ reshape(ks, (1, TILE_K))
        kmask = kidx .< p1m
        vals = ct.gather(nzval, kidx; mask=kmask)
        cols = ct.gather(colval, kidx; mask=kmask, padding_value=Int32(1))
        bt = ct.gather(B, (reshape(cols, (TILE_M, TILE_K, 1)),
                           reshape(ncols, (1, 1, TILE_N))))
        acc = acc .+ reshape(sum(reshape(vals, (TILE_M, TILE_K, 1)) .* bt; dims=2),
                             (TILE_M, TILE_N))
    end
    cidx = (reshape(rows, (TILE_M, 1)), reshape(ncols, (1, TILE_N)))
    out = alpha .* acc
    if BETA_NZ
        out = out .+ beta .* ct.gather(C, cidx)
    end
    ct.scatter(C, cidx, out)
    return
end

"""
    build_spmm(T; tile_m, tile_n, tile_k, beta_nz) -> spmm!

Compile a CSR SpMM kernel for element type `T` and return a launcher
`spmm!(C, rowptr, colval, nzval, B, α, β)`. `tile_m == 1` selects the
row-per-program kernel; `tile_m > 1` the TILE_M-rows-per-program one.
`beta_nz == false` builds the specialization that never reads `C`
(required when C is uninitialized and β == 0).
"""
function build_spmm(::Type{T}; tile_m::Int, tile_n::Int, tile_k::Int,
                    beta_nz::Bool) where {T}
    if tile_m == 1
        k = TritonRun.triton_kernel(spmm_csr_row_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta1(T),
                  spmm_ta2(T), T, T, ct.Constant{Int, tile_n},
                  ct.Constant{Int, tile_k}, ct.Constant{Bool, beta_nz}};
            name="spmm_csr_row", num_warps=4)
        return (C, rp, cv, nz, B, α, β) ->
            TritonRun.launch!(k, (size(C, 1), cld(size(C, 2), tile_n)),
                              C, rp, cv, nz, B, T(α), T(β))
    else
        k = TritonRun.triton_kernel(spmm_csr_rows_kernel,
            Tuple{spmm_ta2(T), spmm_ta1(Int32), spmm_ta1(Int32), spmm_ta1(T),
                  spmm_ta2(T), T, T, ct.Constant{Int, tile_m},
                  ct.Constant{Int, tile_n}, ct.Constant{Int, tile_k},
                  ct.Constant{Bool, beta_nz}};
            name="spmm_csr_rows", num_warps=4)
        return (C, rp, cv, nz, B, α, β) ->
            TritonRun.launch!(k, spmm_grid(C, tile_m, tile_n),
                              C, rp, cv, nz, B, T(α), T(β))
    end
end

# Candidate (tile_m, tile_n, tile_k) configs; tile_m == 1 is the
# row-per-program kernel. Register footprint capped at 4096 gathered
# B elements per program.
function csr_tile_candidates(n)
    tn = clamp(nextpow(2, n), 4, 64)
    cands = [(1, tn, 32), (1, tn, 64)]
    for tm in (8, 32), tk in (16, 32)
        tm * tn * tk <= 4096 && push!(cands, (tm, tn, tk))
    end
    return cands
end
