import cuTile as ct

struct SplitRangeCSRMatrix{T, Ti, V1, V2, Vi1, Vi2, Vi3} <: AbstractMatrix{T}
    contiguous_lo::Vi1
    contiguous_hi::Vi1
    contiguous_vals::V1
    scattered_ptr::Vi2
    scattered_inds::Vi3
    scattered_vals::V2
end
SplitRangeCSRMatrix(ch::Vi1, cl::Vi1, cv::V1, sp::Vi2, si::Vi3, sv::V2) where {V1, V2, Vi1, Vi2, Vi3} =
    SplitRangeCSRMatrix{eltype(V1), eltype(Vi1), V1, V2, Vi1, Vi2, Vi3}(ch, cl, cv, sp, si, sv)

Adapt.@adapt_structure SplitRangeCSRMatrix
Base.size(A::SplitRangeCSRMatrix) = (length(A.contiguous_lo), length(A.contiguous_vals))

function spmm_rc_kernel(C, A::SplitRangeCSRMatrix{T, I}, B, α, β, TILE_M, TILE_N, TILE_K, BETA_NZ, ::Type{ACC} = eltype(C)) where {T, I, ACC}
    (; contiguous_lo, contiguous_hi, contiguous_vals, scattered_ptr, scattered_inds, scattered_vals) = A
    m, n = ct.bid(1), ct.bid(2)
    row_indices = (m - I(1)) * I(TILE_M) .+ ct.arange(TILE_M)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    contiguous_start = ct.gather(contiguous_lo, row_indices)
    contiguous_offset = contiguous_start .- I(1)
    contiguous_length = ct.gather(contiguous_hi, row_indices) .- contiguous_start

    scattered_start = ct.gather(scattered_ptr, row_indices)
    scattered_offset = scattered_start .- I(1)
    scattered_length = ct.gather(scattered_ptr, row_indices .+ I(1)) .- scattered_start

    total_iters = cld(maximum(contiguous_length .+ scattered_length), I(TILE_K))
    acc = zeros(ACC, TILE_M, TILE_N)

    for t in I(1):total_iters
        k = (t - I(1)) * I(TILE_K) .+ k₀

        contiguous_mask = k .≤ contiguous_length
        contiguous_i = contiguous_offset .+ k
        contiguous_v = ct.gather(contiguous_vals, contiguous_i; mask = contiguous_mask)

        scattered_ptrs = scattered_offset .+ (k .- contiguous_length)
        scattered_mask = contiguous_length .< k .≤ contiguous_length .+ scattered_length
        scattered_i = ct.gather(scattered_inds, scattered_ptrs; mask = scattered_mask)
        scattered_v = ct.gather(scattered_vals, scattered_ptrs; mask = scattered_mask)

        i = ifelse.(contiguous_mask, contiguous_i, scattered_i)
        v = convert(ct.Tile{ACC}, contiguous_v .+ scattered_v)
        b_vals = convert(ct.Tile{ACC}, ct.gather(B, (i, col_indices)))
        dot = dropdims(sum(v .* b_vals; dims = 2); dims = 2)

        acc = acc .+ dot
    end

    res = ACC(α) .* acc
    if BETA_NZ
        res = res .+ ACC(β) .* convert(ct.Tile{ACC}, ct.load(C, (m, n), (TILE_M, TILE_N)))
    end
    ct.store(C, (m, n), convert(ct.Tile{eltype(C)}, res))

    return nothing
end

