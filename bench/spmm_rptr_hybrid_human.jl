struct HybridSparseMatrix{T, Ti, M <: SplitRangeCSRMatrix{T, Ti}, V, Vi} <: AbstractMatrix{T}
    nice_part::M
    odd_part::Dict{Ti, Tuple{Vi, V}}
end

function spmm_hybrid!(C, A::HybridSparseMatrix{T, Ti}, B, α, β, todo...) where {T, Ti}
    ct.launch(spmm_rc_kernel, grid, C, A.nice_part, B, α, β,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk),
              ct.Constant(false))

    tmp = CuArray{T}(undef, 1, size(B, 2))
    for (i, (row_indices, values)) in A.odd_part
        sum!(tmp, Base.broadcasted(*, view(B, row_indices, :), values))
        C[i, :] .+= α .* tmp
    end
    return C
end

