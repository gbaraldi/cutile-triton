using CUDA.CUDACore.GPUArrays

struct HybridSparseMatrix{T, Ti, M <: SplitRangeCSRMatrix{T, Ti}, V, Vi} <: AbstractMatrix{T}
    nice_part::M
    odd_part::Dict{Ti, Tuple{Vi, V}}
end
Base.size(A::HybridSparseMatrix) = size(A.nice_part)

# use wide enough type in TileArray construction, otherwise we get crashes on large instances
function launch_rc!(C, A::SplitRangeCSRMatrix{T, Ti}, B, α, β, tm, tn, tk, beta_nz) where {T, Ti}
    grid = (cld(size(C, 1), tm), cld(size(C, 2), tn))
    wrap(X) = ct.TileArray(X; index=length(X) > typemax(Int32) ? Int64 : Ti)
    ct.launch(spmm_rc_kernel, grid, wrap(C), A, wrap(B), α, β,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk), ct.Constant(beta_nz))
end

function spmm_hybrid!(C, A::HybridSparseMatrix{T, Ti}, B, α, β, tm, tn, tk, beta_nz) where {T, Ti}
    launch_rc!(C, A.nice_part, B, α, β, tm, tn, tk, beta_nz)

    tmp = CuArray{T}(undef, 1, size(B, 2))
    for (i, (row_indices, values)) in A.odd_part
        GPUArrays.mapreducedim!(
            identity, +, tmp,
            Base.broadcasted(*, view(B, row_indices, :), values);
            init = zero(T),
        )
        C[i:i, :] .+= α .* tmp
    end
    return C
end
