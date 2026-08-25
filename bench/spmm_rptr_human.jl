import cuTile as ct
using CUDA
using CUDA: i32

struct SplitRangeCSRMatrix{T, Ti, V <: AbstractVector{T}, Vi <: AbstractVector{Ti}} <: AbstractMatrix{T}
    contiguous_ptr::Vi
    contiguous_vals::V
    scattered_ptr::Vi
    scattered_inds::Vi
    scattered_vals::V
end

function spmm_rc_kernel(C, A::SplitRangeCSRMatrix{T, I}, B, α, β, TILE_M, TILE_N, TILE_K, BETA_NZ) where {T, I}
    (; contiguous_ptr, contiguous_vals, scattered_ptr, scattered_inds, scattered_vals) = A
    m, n = ct.bid(1), ct.bid(2)
    row_indices = (m - I(1)) * TILE_M .+ ct.arange(TILE_M)

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
        contiguous_acc = ct.gather(contiguous_vals, contiguous_offset .+ k; mask = contiguous_mask)

        scattered_ptrs = scattered_offset .+ (k .- contiguous_length)
        scattered_mask = contiguous_length .< k .≤ contiguous_length .+ scattered_length
        scattered_acc = ct.gather(scattered_vals, scattered_ptrs; mask = scattered_mask)

        acc = acc .+ contiguous_acc .+ scattered_acc
    end

    res = α .* acc
    if BETA_NZ
        res = res .+ β .* ct.load(C, (m, n), (TILE_M, TILE_N))
    end
    ct.store!(C, (m, n), res)
end

