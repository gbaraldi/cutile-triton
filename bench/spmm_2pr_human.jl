import cuTile as ct

struct TwoPerRowMatrix{T, Ti, V, Vi} <: AbstractMatrix{T}
    inds::Vi
    vals::V
end
TwoPerRowMatrix(inds::Vi, vals::V) where {V, Vi} =
    TwoPerRowMatrix{eltype(V), eltype(Vi), V, Vi}(inds, vals)

Adapt.@adapt_structure TwoPerRowMatrix

function spmm_2pr_kernel(C, A::TwoPerRowMatrix{T, I}, B, α, β, TILE_M, TILE_N, BETA_NZ, ::Type{ACC} = eltype(C)) where {T, I, ACC}
    (; inds, vals) = A
    m, n = ct.bid(1), ct.bid(2)
    col_indices = reshape((n - I(1)) * I(TILE_N) .+ ct.arange(TILE_N), 1, 1, TILE_N)

    i = ct.load(inds, (1, m), (2, TILE_M))
    vals = convert(ct.Tile{ACC}, ct.load(vals, (1, m), (2, TILE_M)))

    b_vals = convert(ct.Tile{ACC}, ct.gather(B, (i, col_indices)))
    res = ACC(α) .* dropdims(sum(vals .* b_vals; dims = 1); dims = 1)
    if BETA_NZ
        res = res .+ ACC(β) .* convert(ct.Tile{ACC}, ct.load(C, (m, n), (TILE_M, TILE_N)))
    end
    ct.store(C, (m, n), convert(ct.Tile{eltype(C)}, res))

    return nothing
end
