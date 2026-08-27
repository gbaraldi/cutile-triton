using CUDA.cuSPARSE: CuSparseMatrixCSR

struct HybridSparseMatrix{T, Ti, M <: SplitRangeCSRMatrix{T, Ti}, S, Vi} <: AbstractMatrix{T}
    nice_part::M
    odd_part::S       # nodd × k sparse matrix of the heavy rows (CuSparseMatrixCSR)
    odd_rows::Vi      # their row indices in C
end
Base.size(A::HybridSparseMatrix) = size(A.nice_part)

# use wide enough type in TileArray construction, otherwise we get crashes on large instances
function launch_rc!(C, A::SplitRangeCSRMatrix{T, Ti}, B, α, β, tm, tn, tk, beta_nz) where {T, Ti}
    grid = (cld(size(C, 1), tm), cld(size(C, 2), tn))
    wrap(X) = ct.TileArray(X; index=length(X) > typemax(Int32) ? Int64 : Ti)
    ct.launch(spmm_rc_kernel, grid, wrap(C), A, wrap(B), α, β,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk), ct.Constant(beta_nz))
end

function spmm_hybrid!(C, A::HybridSparseMatrix{T}, B, α, β, tm, tn, tk, beta_nz) where {T}
    launch_rc!(C, A.nice_part, B, α, β, tm, tn, tk, beta_nz)
    tmp = CuMatrix{T}(undef, size(A.odd_part, 1), size(B, 2))
    mul!(tmp, A.odd_part, B, α, zero(T))
    C[A.odd_rows, :] .+= tmp
    return C
end
