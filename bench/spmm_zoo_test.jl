# Correctness smoke test for bench/spmm_zoo_kernels.jl on small synthetic
# matrices with awkward (non-tile-multiple) sizes. CUDA only.
#
#   julia --project=. bench/spmm_zoo_test.jl

using LinearAlgebra, SparseArrays, Random
using CUDA
using TileTriton
using TileTriton.TritonRun
import cuTile as ct

include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

const T = Float32
rng = MersenneTwister(7)
rtol = sqrt(eps(T)) * 100

function check(name, Cd, Cref)
    ok = isapprox(Array(Cd), Cref; rtol)
    println(ok ? "PASS" : "FAIL", "\t", name,
            ok ? "" : " (max abs err $(maximum(abs.(Array(Cd) .- Cref))))")
    return ok
end

allok = true

# --- Matrix2PerRow{PM}: 2×m colidx, 0 = absent ------------------------------
let m = 1003, k = 517, n = 37
    global allok
    colidx = zeros(Int32, 2, m)
    for i in 1:m
        rand(rng, Bool) && (colidx[1, i] = rand(rng, 1:k))
        rand(rng, Bool) && (colidx[2, i] = rand(rng, 1:k))
    end
    vals = randn(rng, T, 2, m)
    I, J, Vpm, Vv = Int32[], Int32[], T[], T[]
    for i in 1:m, s in 1:2
        j = colidx[s, i]
        j > 0 || continue
        push!(I, i); push!(J, j)
        push!(Vpm, s == 1 ? one(T) : -one(T))
        push!(Vv, vals[s, i])
    end
    Apm = sparse(I, J, Vpm, m, k)
    Av = sparse(I, J, Vv, m, k)
    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    α, β = T(2.5), T(-0.5)
    dcol = CuArray(vec(colidx))
    dvals = CuArray(vec(vals))
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n)

    spmm! = build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=false)
    fill!(Cd, T(NaN))
    spmm!(Cd, dcol, Bd, 1, 0)
    allok &= check("2pr pm β=0", Cd, Apm * Bh)
    spmm_ab! = build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=true)
    copyto!(Cd, C0h)
    spmm_ab!(Cd, dcol, Bd, α, β)
    allok &= check("2pr pm αβ", Cd, α .* (Apm * Bh) .+ β .* C0h)

    spmm! = build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false, beta_nz=false)
    fill!(Cd, T(NaN))
    spmm!(Cd, dcol, dvals, Bd, 1, 0)
    allok &= check("2pr vals β=0", Cd, Av * Bh)
    spmm_ab! = build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false, beta_nz=true)
    copyto!(Cd, C0h)
    spmm_ab!(Cd, dcol, dvals, Bd, α, β)
    allok &= check("2pr vals αβ", Cd, α .* (Av * Bh) .+ β .* C0h)
end

# --- JDSMatrix{PM}: rows sorted by decreasing length ------------------------
let m = 1003, k = 517, n = 37
    global allok
    rowlens = sort!(rand(rng, 0:9, m); rev=true)
    maxlen = rowlens[1]
    colidx = Int32[]
    nzval = T[]
    iterptr = Int32[1]
    for j in 1:maxlen
        cnt = count(>=(j), rowlens)
        for _ in 1:cnt
            push!(colidx, rand(rng, Bool) ? rand(rng, 1:k) : -rand(rng, 1:k))
            push!(nzval, randn(rng, T))
        end
        push!(iterptr, Int32(length(colidx) + 1))
    end
    I, J, Vpm, Vv = Int32[], Int32[], T[], T[]
    for j in 1:maxlen, i in 1:(iterptr[j + 1] - iterptr[j])
        ptr = iterptr[j] + i - 1
        kk = colidx[ptr]
        push!(I, i); push!(J, abs(kk))
        push!(Vpm, kk < 0 ? -one(T) : one(T))
        push!(Vv, nzval[ptr])
    end
    Apm = sparse(I, J, Vpm, m, k)
    Av = sparse(I, J, Vv, m, k)
    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    α, β = T(2.5), T(-0.5)
    dcol = CuArray(colidx)
    dnz = CuArray(nzval)
    diter = CuArray(push!(copy(iterptr), iterptr[end]))   # trailing sentinel
    dcol_abs = CuArray(abs.(colidx))
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n)

    spmm! = build_spmm_jds(T; tile_m=32, tile_n=16, pm=true, beta_nz=false)
    fill!(Cd, T(NaN))
    spmm!(Cd, dcol, diter, Bd, 1, 0)
    allok &= check("jds pm β=0", Cd, Apm * Bh)
    spmm_ab! = build_spmm_jds(T; tile_m=32, tile_n=16, pm=true, beta_nz=true)
    copyto!(Cd, C0h)
    spmm_ab!(Cd, dcol, diter, Bd, α, β)
    allok &= check("jds pm αβ", Cd, α .* (Apm * Bh) .+ β .* C0h)

    spmm! = build_spmm_jds(T; tile_m=64, tile_n=8, pm=false, beta_nz=false)
    fill!(Cd, T(NaN))
    spmm!(Cd, dcol_abs, diter, dnz, Bd, 1, 0)
    allok &= check("jds vals β=0", Cd, Av * Bh)
    spmm_ab! = build_spmm_jds(T; tile_m=64, tile_n=8, pm=false, beta_nz=true)
    copyto!(Cd, C0h)
    spmm_ab!(Cd, dcol_abs, diter, dnz, Bd, α, β)
    allok &= check("jds vals αβ", Cd, α .* (Av * Bh) .+ β .* C0h)
end

println(allok ? "ZOO TEST OK" : "ZOO TEST FAILED")
exit(allok ? 0 : 1)
