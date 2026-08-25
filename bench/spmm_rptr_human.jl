import cuTile as ct
using CUDA
using CUDA: i32

struct SplitRangeCSRMatrix{T, Ti, V <: AbstractVector{T}, Vi1 <: AbstractVector{Ti}, Vi2 <: AbstractVector{Ti}} <: AbstractMatrix{T}
    contiguous_ptr::Vi1
    contiguous_vals::V
    scattered_ptr::Vi1
    scattered_inds::Vi2
    scattered_vals::V
end

Adapt.@adapt_structure SplitRangeCSRMatrix
Base.size(A::SplitRangeCSRMatrix) = (length(A.contiguous_ptr) - 1, length(A.contiguous_vals))

function spmm_rc_kernel(C, A::SplitRangeCSRMatrix{T, I}, B, α, β, TILE_M, TILE_N, TILE_K, BETA_NZ) where {T, I}
    (; contiguous_ptr, contiguous_vals, scattered_ptr, scattered_inds, scattered_vals) = A
    m, n = ct.bid(1), ct.bid(2)
    row_indices = (m - I(1)) * TILE_M .+ ct.arange(TILE_M)
    col_indices = reshape((n - I(1)) * TILE_N .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    contiguous_start = ct.gather(contiguous_ptr, row_indices)
    contiguous_offset = contiguous_start .- I(1)
    contiguous_length = ct.gather(contiguous_ptr, row_indices .+ I(1)) .- contiguous_start

    scattered_start = ct.gather(scattered_ptr, row_indices)
    scattered_offset = scattered_start .- I(1)
    scattered_length = ct.gather(scattered_ptr, row_indices .+ I(1)) .- scattered_start

    total_iters = cld(maximum(contiguous_length .+ scattered_length), TILE_K)
    acc = zeros(T, TILE_M, TILE_N)

    for t in I(1):total_iters
        k = (t - I(1)) * TILE_K .+ k₀

        contiguous_mask = k .≤ contiguous_length
        contiguous_i = contiguous_offset .+ k
        contiguous_vals = ct.gather(contiguous_vals, contiguous_i; mask = contiguous_mask, padding_value = zero(T))

        scattered_ptrs = scattered_offset .+ (k .- contiguous_length)
        scattered_mask = contiguous_length .< k .≤ contiguous_length .+ scattered_length
        scattered_i = ct.gather(scattered_inds, scattered_ptrs; mask = scattered_mask, padding_value = I(0))
        scattered_vals = ct.gather(scattered_vals, scattered_ptrs; mask = scattered_mask, padding_value = zero(T))

        i = ifelse.(contiguous_mask, contiguous_i, scattered_i)
        b_vals = ct.gather(B, (reshape(i, TILE_M, TILE_K), col_indices))
        dot = dropdims(sum((contiguous_vals .+ scattered_vals) .* b_vals; dims = 2); dims = 2)

        acc = acc .+ dot
    end

    res = α .* acc
    if BETA_NZ
        res = res .+ β .* ct.load(C, (m, n), (TILE_M, TILE_N))
    end
    ct.store(C, (m, n), res)
end

