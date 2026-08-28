import cuTile as ct
using CUDA
using CUDA: i32

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

function spmm_rc_kernel(C, A::SplitRangeCSRMatrix{T, I}, B, α, β, TILE_M, TILE_N, TILE_K, BETA_NZ) where {T, I}
    (; contiguous_lo, contiguous_hi, contiguous_vals, scattered_ptr, scattered_inds, scattered_vals) = A
    m, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * TILE_N .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    contiguous_start = ct.eachtile(contiguous_lo, (TILE_M,); padding_mode = ct.PaddingMode.Zero)[m]
    contiguous_offset = contiguous_start .- I(1)
    contiguous_length = ct.eachtile(contiguous_hi, (TILE_M,); padding_mode = ct.PaddingMode.Zero)[m] .- contiguous_start

    scattered_start = ct.eachtile(scattered_ptr, (TILE_M,); padding_mode = ct.PaddingMode.Zero)[m]
    scattered_offset = scattered_start .- I(1)
    scattered_length = ct.eachtile(scattered_ptr, (TILE_M,); step = (1,), padding_mode = ct.PaddingMode.Zero)[(m - I(1)) * TILE_M + I(2)] .- scattered_start

    total_iters = cld(maximum(contiguous_length .+ scattered_length), I(TILE_K))
    acc = zeros(T, TILE_M, TILE_N)

    for t in I(1):total_iters
        k = (t - I(1)) * I(TILE_K) .+ k₀

        contiguous_mask = k .≤ contiguous_length
        contiguous_i = contiguous_offset .+ k
        contiguous_v = ct.gather(contiguous_vals, contiguous_i; mask = contiguous_mask, padding_value = zero(T))

        scattered_ptrs = scattered_offset .+ (k .- contiguous_length)
        scattered_mask = contiguous_length .< k .≤ contiguous_length .+ scattered_length
        scattered_i = ct.gather(scattered_inds, scattered_ptrs; mask = scattered_mask, padding_value = I(0))
        scattered_v = ct.gather(scattered_vals, scattered_ptrs; mask = scattered_mask, padding_value = zero(T))

        i = ifelse.(contiguous_mask, contiguous_i, scattered_i)
        b_vals = ct.gather(B, (i, col_indices))
        dot = dropdims(sum((contiguous_v .+ scattered_v) .* b_vals; dims = 2); dims = 2)

        acc = acc .+ dot
    end

    res = α .* acc
    if BETA_NZ
        res = res .+ β .* ct.eachtile(C, (TILE_M, TILE_N))[m, n]
    end
    ct.eachtile(C, (TILE_M, TILE_N))[m, n] = res

    return nothing
end

