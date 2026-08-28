import cuTile as ct

# Heavy rows of a SplitRangeCSRMatrix, split into fixed-size chunks so that one
# very long row is spread over many programs. `chunk_row[c]` is the (local)
# heavy row that chunk `c` belongs to, `chunk_ptr[h]:chunk_ptr[h+1]-1` are the
# chunks of heavy row `h`, and `row[h]` is its row index in the full matrix.
# `A` holds the heavy rows only, in the same range + scattered layout as the
# light rows.
struct HeavyRowChunks{T, Ti, M <: SplitRangeCSRMatrix{T, Ti}, Vi1, Vi2, Vi3}
    A::M
    row::Vi1
    chunk_row::Vi2
    chunk_ptr::Vi3
end

Adapt.@adapt_structure HeavyRowChunks

# Pass 1: partial[c, :] = A[h, :] * B[:, cols] restricted to the entries of chunk c.
function spmm_heavy_kernel(partial, H::HeavyRowChunks{T, I}, B, TILE_N, TILE_K, CHUNK) where {T, I}
    (; contiguous_lo, contiguous_hi, contiguous_vals, scattered_ptr, scattered_inds, scattered_vals) = H.A
    c, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    h = H.chunk_row[c]
    chunk_offset = (c - H.chunk_ptr[h]) * I(CHUNK)

    contiguous_start = contiguous_lo[h]
    contiguous_offset = contiguous_start - I(1)
    contiguous_length = contiguous_hi[h] - contiguous_start

    scattered_start = scattered_ptr[h]
    scattered_offset = scattered_start - I(1)
    scattered_length = scattered_ptr[h + I(1)] - scattered_start

    acc = zeros(T, 1, TILE_N)

    for t in I(1):I(CHUNK ÷ TILE_K)
        k = chunk_offset + (t - I(1)) * I(TILE_K) .+ k₀

        contiguous_mask = k .≤ contiguous_length
        contiguous_i = contiguous_offset .+ k
        contiguous_v = ct.gather(contiguous_vals, contiguous_i; mask = contiguous_mask)

        scattered_ptrs = scattered_offset .+ (k .- contiguous_length)
        scattered_mask = contiguous_length .< k .≤ contiguous_length + scattered_length
        scattered_i = ct.gather(scattered_inds, scattered_ptrs; mask = scattered_mask)
        scattered_v = ct.gather(scattered_vals, scattered_ptrs; mask = scattered_mask)

        i = ifelse.(contiguous_mask, contiguous_i, scattered_i)
        b_vals = ct.gather(B, (i, col_indices))
        dot = dropdims(sum((contiguous_v .+ scattered_v) .* b_vals; dims = 2); dims = 2)

        acc = acc .+ dot
    end

    ct.store(partial, (c, n), acc)

    return nothing
end

# Pass 2: C[row[h], cols] += α * sum(partial[chunk_ptr[h]:chunk_ptr[h+1]-1, cols]; dims = 1).
# The light-row kernel has already written α*0 + β*C for the heavy rows.
function spmm_heavy_reduce_kernel(C, partial, H::HeavyRowChunks{T, I}, α, TILE_C, TILE_N) where {T, I}
    h, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, TILE_N)

    c₀ = ct.arange(TILE_C)

    chunk_start = H.chunk_ptr[h]
    chunk_offset = chunk_start - I(1)
    chunk_length = H.chunk_ptr[h + I(1)] - chunk_start

    total_iters = cld(chunk_length, I(TILE_C))
    acc = zeros(T, TILE_C, TILE_N)

    for t in I(1):total_iters
        c = chunk_offset + (t - I(1)) * I(TILE_C) .+ c₀
        mask = c .≤ chunk_offset + chunk_length
        acc = acc .+ ct.gather(partial, (c, col_indices); mask)
    end

    row = ct.Tile(H.row[h])
    res = ct.gather(C, (row, col_indices)) .+ α .* sum(acc; dims = 1)
    ct.scatter(C, (row, col_indices), res)

    return nothing
end
