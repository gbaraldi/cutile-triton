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
const m, k, n = 1003, 517, 37
const rtol = sqrt(eps(T)) * 100

function check(name, Cd, Cref)
    ok = isapprox(Array(Cd), Cref; rtol)
    println(ok ? "PASS" : "FAIL", "\t", name,
            ok ? "" : " (max abs err $(maximum(abs.(Array(Cd) .- Cref))))")
    return ok
end

"Build the β=0 and general-α/β specializations of one kernel via
`build(beta_nz)`, run both through `launch(spmm!, C, α, β)`, check both.
`tr` marks the fully transposed layout: Cd, the launched B, and the
references are all transposed."
function run_pair(name, build, launch, Cd, Aref, Bh, C0h, α, β; tr=false)
    Cref = Aref * Bh
    tr && (Cref = permutedims(Cref); C0h = permutedims(C0h))
    spmm! = build(false)
    fill!(Cd, T(NaN))
    launch(spmm!, Cd, 1, 0)
    ok = check("$name β=0", Cd, Cref)
    spmm_ab! = build(true)
    copyto!(Cd, C0h)
    launch(spmm_ab!, Cd, α, β)
    return ok & check("$name αβ", Cd, α .* Cref .+ β .* C0h)
end

# --- Matrix2PerRow{PM}: 2×m colidx, 0 = absent ------------------------------
function test_2pr(rng)
    colidx = zeros(Int32, 2, m)
    for i in 1:m, s in 1:2
        rand(rng, Bool) && (colidx[s, i] = rand(rng, 1:k))
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
    dcol = CuArray(colidx)
    dvals = CuArray(vals)
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n)

    Btd = CuArray(permutedims(Bh))
    Cdt = CuArray{T}(undef, n, m)

    ok = run_pair("2pr pm",
                  bnz -> build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dcol, Bd, α, β),
                  Cd, Apm, Bh, C0h, α, β)
    ok &= run_pair("2pr vals",
                  bnz -> build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dcol, dvals, Bd, α, β),
                  Cd, Av, Bh, C0h, α, β)
    ok &= run_pair("2pr pm t",
                  bnz -> build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true,
                                        beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dcol, Btd, α, β),
                  Cdt, Apm, Bh, C0h, α, β; tr=true)
    ok & run_pair("2pr vals t",
                  bnz -> build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false,
                                        beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dcol, dvals, Btd, α, β),
                  Cdt, Av, Bh, C0h, α, β; tr=true)
end

# --- JDSMatrix{PM}: rows sorted by decreasing length ------------------------
function test_jds(rng)
    rowlens = sort!(rand(rng, 0:9, m); rev=true)
    maxlen = rowlens[1]
    colidx = Int32[]
    nzval = T[]
    iterptr = Int32[1]
    for j in 1:maxlen
        for _ in 1:count(>=(j), rowlens)
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
        push!(Vpm, flipsign(one(T), kk))
        push!(Vv, nzval[ptr])
    end
    Apm = sparse(I, J, Vpm, m, k)
    Av = sparse(I, J, Vv, m, k)
    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    α, β = T(2.5), T(-0.5)
    dcol = CuArray(colidx)
    dcol_abs = CuArray(abs.(colidx))
    dnz = CuArray(nzval)
    diter = CuArray(iterptr)
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n)

    Btd = CuArray(permutedims(Bh))
    Cdt = CuArray{T}(undef, n, m)

    ok = run_pair("jds pm",
                  bnz -> build_spmm_jds(T; tile_m=32, tile_n=16, pm=true, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dcol, diter, Bd, α, β),
                  Cd, Apm, Bh, C0h, α, β)
    ok &= run_pair("jds vals",
                  bnz -> build_spmm_jds(T; tile_m=64, tile_n=8, pm=false, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dcol_abs, diter, dnz, Bd, α, β),
                  Cd, Av, Bh, C0h, α, β)
    ok &= run_pair("jds pm t",
                  bnz -> build_spmm_jds(T; tile_m=32, tile_n=16, pm=true,
                                        beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dcol, diter, Btd, α, β),
                  Cdt, Apm, Bh, C0h, α, β; tr=true)
    ok & run_pair("jds vals t",
                  bnz -> build_spmm_jds(T; tile_m=64, tile_n=8, pm=false,
                                        beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dcol_abs, diter, dnz, Btd, α, β),
                  Cdt, Av, Bh, C0h, α, β; tr=true)
end

# --- range + CSR: contiguous range of one sign + scattered columns -----------
function test_rc(rng)
    lo = Int32[rand(rng, 1:k) for _ in 1:m]
    hi = Int32[min(l + rand(rng, 0:4), k + 1) for l in lo]     # hi exclusive, may be empty
    inptr = Int32[1]
    inids = Int32[]
    for i in 1:m
        for _ in 1:rand(rng, 0:5)
            push!(inids, rand(rng, 1:k))
        end
        push!(inptr, Int32(length(inids) + 1))
    end
    rsign = -one(T)
    rvals = randn(rng, T, k)            # per column: each column in one range
    invals = randn(rng, T, length(inids))
    I, J, Vpm, Vv = Int32[], Int32[], T[], T[]
    for i in 1:m
        for a in lo[i]:(hi[i] - 1)
            push!(I, i); push!(J, a); push!(Vpm, rsign); push!(Vv, rvals[a])
        end
        for p in inptr[i]:(inptr[i + 1] - 1)
            push!(I, i); push!(J, inids[p]); push!(Vpm, -rsign); push!(Vv, invals[p])
        end
    end
    Apm = sparse(I, J, Vpm, m, k)
    Av = sparse(I, J, Vv, m, k)
    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    α, β = T(2.5), T(-0.5)
    dlo, dhi, dinptr, dinids = CuArray.((lo, hi, inptr, inids))
    drvals, dinvals = CuArray(rvals), CuArray(invals)
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n)
    Btd = CuArray(permutedims(Bh))
    Cdt = CuArray{T}(undef, n, m)

    ok = run_pair("rc pm",
                  bnz -> build_spmm_rc(T; tile_m=32, tile_n=16, tile_k=1, pm=true, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dlo, dhi, dinptr, dinids, rsign, Bd, α, β),
                  Cd, Apm, Bh, C0h, α, β)
    ok &= run_pair("rc vals",
                  bnz -> build_spmm_rc(T; tile_m=16, tile_n=8, tile_k=2, pm=false, beta_nz=bnz),
                  (f!, C, α, β) -> f!(C, dlo, dhi, drvals, dinptr, dinids, dinvals, Bd, α, β),
                  Cd, Av, Bh, C0h, α, β)
    ok &= run_pair("rc pm t",
                  bnz -> build_spmm_rc(T; tile_m=32, tile_n=16, tile_k=4, pm=true,
                                       beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dlo, dhi, dinptr, dinids, rsign, Btd, α, β),
                  Cdt, Apm, Bh, C0h, α, β; tr=true)
    ok & run_pair("rc vals t",
                  bnz -> build_spmm_rc(T; tile_m=16, tile_n=8, tile_k=2, pm=false,
                                       beta_nz=bnz, bt=true),
                  (f!, C, α, β) -> f!(C, dlo, dhi, drvals, dinptr, dinids, dinvals, Btd, α, β),
                  Cdt, Av, Bh, C0h, α, β; tr=true)
end

rng = MersenneTwister(7)
allok = test_2pr(rng) & test_jds(rng) & test_rc(rng)
println(allok ? "ZOO TEST OK" : "ZOO TEST FAILED")
exit(allok ? 0 : 1)
