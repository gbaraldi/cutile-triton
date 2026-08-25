# Does collapsing rc's lo/hi into a single rptr[m+1] array matter?
#
# The ranges of the incidence matrix partition the columns (every arc has
# exactly one head), so in node order they chain — hi[i] == lo[i+1] — and the
# two bounds arrays collapse into one pointer array (= the adjacency matrix's
# CSC colptr). The exported A is in JDS (length-sorted) row order where the
# ranges don't chain, so this benches the KA rc pm kernel in three variants:
#
#   lo/hi jds     the current format on the matrix as exported (baseline)
#   lo/hi node    two arrays, rows permuted to node order — isolates the
#                 row-order effect from the array-merge effect
#   rptr node     the single collapsed array, same node order
#
#   julia --project=. bench/spmm_rptr.jl [n...]      (default n = 8, 64, 256)
#   SPMM_DATA / SPMM_BACKEND / SPMM_TYPE as in bench/spmm_flow.jl.
#
# β=0 path only (the headline configuration); both layouts, nb ∈ {1,4,8}.

using LinearAlgebra, SparseArrays, Random, Printf, Serialization
using SIMD

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_formats.jl"))

# nb_load / nb_store! / ka_rowcol / spmm_rc_pm_ka! as in bench/spmm_flow.jl
@inline nb_load(b, col, b0, ::Val{NB}, ::Val{true}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[b0 + q, col]), Val(NB)))
@inline nb_load(b, col, b0, ::Val{NB}, ::Val{false}) where {NB} =
    Vec(ntuple(q -> @inbounds(b[col, b0 + q]), Val(NB)))

@inline function nb_store!(c, s::Vec{NB}, i, b0, α, β,
                           ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    @inbounds for q in 1:NB
        v = α * s[q]
        if BT
            c[b0 + q, i] = BETA_NZ ? v + β * c[b0 + q, i] : v
        else
            c[i, b0 + q] = BETA_NZ ? v + β * c[i, b0 + q] : v
        end
    end
end

@inline ka_rowcol(idx, ::Val{false}) = idx
@inline ka_rowcol(idx, ::Val{true}) = (idx[2], idx[1])
ka_ndrange(m, nblk, bt) = bt ? (nblk, m) : (m, nblk)

@kernel function spmm_rc_pm_ka!(c, lo, hi, inptr, inids, b, α, β, rsign, ::Val{NB},
                                ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in lo[i]:(hi[i] - Int32(1))
            s += nb_load(b, a, b0, Val(NB), Val(BT))
        end
        for p in inptr[i]:(inptr[i + 1] - Int32(1))
            s -= nb_load(b, inids[p], b0, Val(NB), Val(BT))
        end
        nb_store!(c, s, i, b0, α * rsign, β, Val(BT), Val(BETA_NZ))
    end
end

# The single-array variant: row i's range is [rptr[i], rptr[i+1])
@kernel function spmm_rp_pm_ka!(c, rptr, inptr, inids, b, α, β, rsign, ::Val{NB},
                                ::Val{BT}, ::Val{BETA_NZ}) where {NB, BT, BETA_NZ}
    i, blk = ka_rowcol(@index(Global, NTuple), Val(BT))
    b0 = (blk - 1) * NB
    @inbounds begin
        s = zero(Vec{NB, eltype(c)})
        for a in rptr[i]:(rptr[i + 1] - Int32(1))
            s += nb_load(b, a, b0, Val(NB), Val(BT))
        end
        for p in inptr[i]:(inptr[i + 1] - Int32(1))
            s -= nb_load(b, inids[p], b0, Val(NB), Val(BT))
        end
        nb_store!(c, s, i, b0, α * rsign, β, Val(BT), Val(BETA_NZ))
    end
end

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = get(ENV, "SPMM_TYPE", "Float32") == "Float64" ? Float64 : Float32
    datadir = joinpath(@__DIR__, "data")
    data = get(ENV, "SPMM_DATA", "flow_TX")

    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$data.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$(data)_t.jls")))
    m, k = size(A)
    println("# spmm_rptr bench: $BACKEND ($(device_name())), $T, $data: ",
            "A = $m×$k with nnz=$(nnz(A)), n = $ns")

    rc = range_csr(At.colptr, At.rowval, At.nzval)
    no = node_order_rptr(rc, k)
    println("# rsign $(rc.rsign), $(sum(rc.lo .== rc.hi)) empty ranges, ",
            "ranges tile 1:$k in node order")
    rsign = T(rc.rsign)

    lo_j = GPUArr(rc.lo); hi_j = GPUArr(rc.hi)
    inptr_j = GPUArr(rc.inptr); inids_j = GPUArr(rc.inids)
    lo_n = GPUArr(no.lo); hi_n = GPUArr(no.hi); rptr_n = GPUArr(no.rptr)
    inptr_n = GPUArr(no.inptr); inids_n = GPUArr(no.inids)

    rng = MersenneTwister(42)
    rtol = sqrt(eps(T)) * 100
    for n in ns
        case = "rptr $data n=$n $T"
        flops = 2.0 * nnz(A) * n
        Bh = rand(rng, T, k, n)
        Crefh = A * Bh
        Crefh_n = Crefh[no.perm, :]
        for (bt, lay) in ((false, ""), (true, " t"))
            Bx = GPUArr(bt ? permutedims(Bh) : Bh)
            Cref_j = GPUArr(bt ? permutedims(Crefh) : Crefh)
            Cref_n = GPUArr(bt ? permutedims(Crefh_n) : Crefh_n)
            Cd = similar(Cref_j)
            rc_pm! = spmm_rc_pm_ka!(KA.get_backend(Cd))
            rp_pm! = spmm_rp_pm_ka!(KA.get_backend(Cd))
            for nb in (1, 4, 8)
                n % nb == 0 || continue
                grid = ka_ndrange(m, n ÷ nb, bt)
                variants = (
                    ("lo/hi jds", Cref_j,
                     C -> rc_pm!(C, lo_j, hi_j, inptr_j, inids_j, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)),
                    ("lo/hi node", Cref_n,
                     C -> rc_pm!(C, lo_n, hi_n, inptr_n, inids_n, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)),
                    ("rptr node", Cref_n,
                     C -> rp_pm!(C, rptr_n, inptr_n, inids_n, Bx, T(1), T(0),
                                 rsign, Val(nb), Val(bt), Val(false); ndrange=grid)))
                for (label, Cref, f0) in variants
                    full = "KA rc pm[nb=$nb$lay] $label"
                    fill!(Cd, T(NaN))
                    f0(Cd)
                    device_sync()
                    check(full, Cd, Cref; rtol) || continue
                    t = timeit(() -> f0(Cd))
                    result_row(case, full, t, flops)
                end
                flush(stdout)
            end
            BACKEND == "cuda" && foreach(CUDA.unsafe_free!, (Bx, Cd, Cref_j, Cref_n))
            Bx = Cd = Cref_j = Cref_n = nothing
            GC.gc()
            BACKEND == "cuda" && CUDA.reclaim()
        end
    end
    println("SPMM RPTR BENCH DONE ($BACKEND)")
end

main()
