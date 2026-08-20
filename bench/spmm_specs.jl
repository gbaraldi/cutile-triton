# Shared TileArray argument specs for the bench SpMM kernels (128-byte
# aligned base pointers, unit stride along dim 1), plus the launch grid all
# tiled SpMM kernels use: one program per TILE_M×TILE_N block of C.
# Included by both spmm_csr_kernels.jl and spmm_zoo_kernels.jl.

import cuTile as ct

const SPMM_SPEC1 = ct.ArraySpec{1}(128, true, (1,), (0,))
const SPMM_SPEC2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
spmm_ta1(T) = ct.TileArray{T, 1, Int32, SPMM_SPEC1}
spmm_ta2(T) = ct.TileArray{T, 2, Int32, SPMM_SPEC2}

spmm_grid(C, tile_m, tile_n) = (cld(size(C, 1), tile_m), cld(size(C, 2), tile_n))

# grid for the transposed layout (C stored n×m; bid(1) still walks M-blocks)
spmm_grid_t(C, tile_m, tile_n) = (cld(size(C, 2), tile_m), cld(size(C, 1), tile_n))
