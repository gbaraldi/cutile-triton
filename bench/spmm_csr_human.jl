import cuTile as ct

# Plain CSR, 1-based: row v owns the entries rowptr[v]:rowptr[v+1]-1 of
# colval/nzval, with the column ids of a row ascending. `ncols` is carried
# alongside because the three arrays do not determine it.
struct CSRMatrix{T, Ti, Vp, Vi, Vv} <: AbstractMatrix{T}
    rowptr::Vp
    colval::Vi
    nzval::Vv
    ncols::Int
end
CSRMatrix(rowptr::Vp, colval::Vi, nzval::Vv, ncols) where {Vp, Vi, Vv} =
    CSRMatrix{eltype(Vv), eltype(Vp), Vp, Vi, Vv}(rowptr, colval, nzval, ncols)

Adapt.@adapt_structure CSRMatrix
Base.size(A::CSRMatrix) = (length(A.rowptr) - 1, A.ncols)

# C = α·A·B + β·C over a TILE_M × TILE_N block of C. The block's rows advance
# together in TILE_K-wide steps to the longest of them, each step gathering
# the rows' column ids and values and then the matching rows of B. Lanes past
# a row's length are masked: their column id gathers as 0, which the B gather
# bounds-masks to a zero row.
function spmm_csr_kernel(C, A::CSRMatrix{T, I}, B, α, β, TILE_M, TILE_N, TILE_K, BETA_NZ, ::Type{ACC} = eltype(C)) where {T, I, ACC}
    (; rowptr, colval, nzval) = A
    m, n = ct.bid(1), ct.bid(2)
    row_indices = (m - I(1)) * I(TILE_M) .+ ct.arange(TILE_M)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    row_start = ct.gather(rowptr, row_indices)
    row_offset = row_start .- I(1)
    row_length = ct.gather(rowptr, row_indices .+ I(1)) .- row_start

    total_iters = cld(maximum(row_length), I(TILE_K))
    acc = zeros(ACC, TILE_M, TILE_N)

    for t in I(1):total_iters
        k = (t - I(1)) * I(TILE_K) .+ k₀

        mask = k .≤ row_length
        ptrs = row_offset .+ k
        i = ct.gather(colval, ptrs; mask)
        v = convert(ct.Tile{ACC}, ct.gather(nzval, ptrs; mask))
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

# Heavy rows of a CSRMatrix, split into fixed-size chunks so that one very
# long row is spread over many programs. `chunk_row[c]` is the (local) heavy
# row that chunk `c` belongs to, `chunk_ptr[h]:chunk_ptr[h+1]-1` are the
# chunks of heavy row `h`, and `row[h]` is its row index in the full matrix.
# `A` holds the heavy rows only, in the same CSR layout as the light rows.
struct HeavyCSRChunks{T, Ti, M <: CSRMatrix{T, Ti}, Vi1, Vi2, Vi3}
    A::M
    row::Vi1
    chunk_row::Vi2
    chunk_ptr::Vi3
end

Adapt.@adapt_structure HeavyCSRChunks

# Pass 1: partial[c, :] = A[h, :] * B[:, cols] restricted to the entries of chunk c.
function spmm_csr_heavy_kernel(partial, H::HeavyCSRChunks{T, I}, B, TILE_N, TILE_K, CHUNK, ::Type{ACC} = eltype(partial)) where {T, I, ACC}
    (; rowptr, colval, nzval) = H.A
    c, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, 1, TILE_N)

    k₀ = reshape(ct.arange(TILE_K), 1, TILE_K)

    h = H.chunk_row[c]
    chunk_offset = (c - H.chunk_ptr[h]) * I(CHUNK)

    row_start = rowptr[h]
    row_offset = row_start - I(1)
    row_length = rowptr[h + I(1)] - row_start

    acc = zeros(ACC, 1, TILE_N)

    for t in I(1):I(CHUNK ÷ TILE_K)
        k = chunk_offset + (t - I(1)) * I(TILE_K) .+ k₀

        mask = k .≤ row_length
        ptrs = row_offset .+ k
        i = ct.gather(colval, ptrs; mask)
        v = convert(ct.Tile{ACC}, ct.gather(nzval, ptrs; mask))
        b_vals = convert(ct.Tile{ACC}, ct.gather(B, (i, col_indices)))
        dot = dropdims(sum(v .* b_vals; dims = 2); dims = 2)

        acc = acc .+ dot
    end

    ct.store(partial, (c, n), acc)

    return nothing
end

# Pass 2: C[row[h], cols] += α * sum(partial[chunk_ptr[h]:chunk_ptr[h+1]-1, cols]; dims = 1).
# The light-row kernel has already written α*0 + β*C for the heavy rows.
function spmm_csr_heavy_reduce_kernel(C, partial, H::HeavyCSRChunks{T, I}, α, TILE_C, TILE_N, ::Type{ACC} = eltype(partial)) where {T, I, ACC}
    h, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, TILE_N)

    c₀ = ct.arange(TILE_C)

    chunk_start = H.chunk_ptr[h]
    chunk_offset = chunk_start - I(1)
    chunk_length = H.chunk_ptr[h + I(1)] - chunk_start

    total_iters = cld(chunk_length, I(TILE_C))
    acc = zeros(ACC, TILE_C, TILE_N)

    for t in I(1):total_iters
        c = chunk_offset + (t - I(1)) * I(TILE_C) .+ c₀
        mask = c .≤ chunk_offset + chunk_length
        acc = acc .+ ct.gather(partial, (c, col_indices); mask)
    end

    row = ct.Tile(H.row[h])
    res = convert(ct.Tile{ACC}, ct.gather(C, (row, col_indices))) .+ ACC(α) .* sum(acc; dims = 1)
    ct.scatter(C, (row, col_indices), convert(ct.Tile{eltype(C)}, res))

    return nothing
end
