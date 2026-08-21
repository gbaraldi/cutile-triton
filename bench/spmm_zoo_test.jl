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
include(joinpath(@__DIR__, "spmm_formats.jl"))

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

# --- exactly two entries per row: unchecked B gathers (needs tile_n | n) -----
function test_2pr_exact2(rng)
    n2 = 32
    colidx = rand(rng, Int32(1):Int32(k), 2, m)
    vals = randn(rng, T, 2, m)
    I = repeat(Int32.(1:m); inner=2)
    Apm = sparse(I, vec(colidx), repeat(T[1, -1], m), m, k)
    Av = sparse(I, vec(colidx), vec(vals), m, k)
    Bh = rand(rng, T, k, n2)
    C0h = rand(rng, T, m, n2)
    α, β = T(2.5), T(-0.5)
    dcol = CuArray(colidx)
    dvals = CuArray(vals)
    Bd = CuArray(Bh)
    Cd = CuArray{T}(undef, m, n2)
    Btd = CuArray(permutedims(Bh))
    Cdt = CuArray{T}(undef, n2, m)

    ok = run_pair("2pr pm x2",
                  bnz -> build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=bnz,
                                        exact2=true),
                  (f!, C, α, β) -> f!(C, dcol, Bd, α, β),
                  Cd, Apm, Bh, C0h, α, β)
    ok &= run_pair("2pr vals x2",
                  bnz -> build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false, beta_nz=bnz,
                                        exact2=true),
                  (f!, C, α, β) -> f!(C, dcol, dvals, Bd, α, β),
                  Cd, Av, Bh, C0h, α, β)
    ok &= run_pair("2pr pm x2 t",
                  bnz -> build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=bnz,
                                        bt=true, exact2=true),
                  (f!, C, α, β) -> f!(C, dcol, Btd, α, β),
                  Cdt, Apm, Bh, C0h, α, β; tr=true)
    ok &= run_pair("2pr vals x2 t",
                  bnz -> build_spmm_2pr(T; tile_m=64, tile_n=8, pm=false, beta_nz=bnz,
                                        bt=true, exact2=true),
                  (f!, C, α, β) -> f!(C, dcol, dvals, Btd, α, β),
                  Cdt, Av, Bh, C0h, α, β; tr=true)
    # tile_n ∤ n must be refused rather than read past B
    bad = build_spmm_2pr(T; tile_m=32, tile_n=16, pm=true, beta_nz=false, exact2=true)
    refused = try
        bad(CuArray{T}(undef, m, n), dcol, CuArray(rand(rng, T, k, n)), 1, 0); false
    catch err
        err isa ArgumentError
    end
    println(refused ? "PASS" : "FAIL", "\t2pr x2 refuses tile_n ∤ n")
    return ok & refused
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

# --- hybrid rc + heavy rows: a few long rows peeled into chunk partials -----
function test_rch(rng)
    lo = Int32[rand(rng, 1:k) for _ in 1:m]
    hi = Int32[min(l + rand(rng, 0:4), k + 1) for l in lo]
    lens = [rand(rng, 0:5) for _ in 1:m]
    lo[5] = 1; hi[5] = 301; lens[5] = 200          # 500 entries: range + scattered
    lo[900] = 7; hi[900] = 7; lens[900] = 150      # scattered only
    lo[m] = 100; hi[m] = 420; lens[m] = 0          # range only, last row
    inptr = Int32[1]
    inids = Int32[]
    for i in 1:m
        for _ in 1:lens[i]
            push!(inids, rand(rng, 1:k))
        end
        push!(inptr, Int32(length(inids) + 1))
    end
    rsign = -one(T)
    rvals = randn(rng, T, k)
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
    rch = split_heavy((; lo, hi, inptr, inids, rsign); maxlen=32)
    l, hv = rch.light, rch.heavy
    @assert hv.row == Int32[5, 900, m]
    chunk = 64
    ch = heavy_chunks(hv; chunk)
    # scattered values follow their entries into the light / heavy CSRs
    isheavy = falses(m); isheavy[hv.row] .= true
    linvals = T[invals[p] for i in 1:m if !isheavy[i] for p in inptr[i]:(inptr[i + 1] - 1)]
    hinvals = T[invals[p] for i in hv.row for p in inptr[i]:(inptr[i + 1] - 1)]
    Bh = rand(rng, T, k, n)
    C0h = rand(rng, T, m, n)
    α, β = T(2.5), T(-0.5)
    dl = map(CuArray, (l.lo, l.hi, l.inptr, l.inids))
    dh = map(CuArray, (hv.row, hv.lo, hv.hi, hv.ptr, hv.ids, ch.crow, ch.cptr))
    drvals, dlinvals, dhinvals = CuArray(rvals), CuArray(linvals), CuArray(hinvals)
    nchunk = length(ch.crow)
    Bd = CuArray(Bh); Cd = CuArray{T}(undef, m, n); part = CuArray{T}(undef, nchunk, n)
    Btd = CuArray(permutedims(Bh)); Cdt = CuArray{T}(undef, n, m); partt = CuArray{T}(undef, n, nchunk)

    function rch_pm(bnz, bt)
        light! = build_spmm_rc(T; tile_m=32, tile_n=16, tile_k=1, pm=true, beta_nz=bnz, bt)
        part!, red! = build_spmm_heavy(T; tile_n=16, tile_k=16, chunk, pm=true, bt, tile_c=8)
        return (C, B, α, β) -> begin
            light!(C, dl..., rsign, B, α, β)
            part!(bt ? partt : part, dh[6], dh[7], dh[2], dh[3], dh[4], dh[5], rsign, B)
            red!(C, bt ? partt : part, dh[1], dh[7], α)
        end
    end
    function rch_vals(bnz, bt)
        light! = build_spmm_rc(T; tile_m=16, tile_n=8, tile_k=2, pm=false, beta_nz=bnz, bt)
        part!, red! = build_spmm_heavy(T; tile_n=8, tile_k=16, chunk, pm=false, bt, tile_c=8)
        return (C, B, α, β) -> begin
            light!(C, dl[1], dl[2], drvals, dl[3], dl[4], dlinvals, B, α, β)
            part!(bt ? partt : part, dh[6], dh[7], dh[2], dh[3], drvals, dh[4], dh[5], dhinvals, B)
            red!(C, bt ? partt : part, dh[1], dh[7], α)
        end
    end
    ok = run_pair("rch pm", bnz -> rch_pm(bnz, false), (f!, C, α, β) -> f!(C, Bd, α, β),
                  Cd, Apm, Bh, C0h, α, β)
    ok &= run_pair("rch vals", bnz -> rch_vals(bnz, false), (f!, C, α, β) -> f!(C, Bd, α, β),
                   Cd, Av, Bh, C0h, α, β)
    ok &= run_pair("rch pm t", bnz -> rch_pm(bnz, true), (f!, C, α, β) -> f!(C, Btd, α, β),
                   Cdt, Apm, Bh, C0h, α, β; tr=true)
    ok & run_pair("rch vals t", bnz -> rch_vals(bnz, true), (f!, C, α, β) -> f!(C, Btd, α, β),
                  Cdt, Av, Bh, C0h, α, β; tr=true)
end

rng = MersenneTwister(7)
allok = test_2pr(rng) & test_2pr_exact2(rng) & test_jds(rng) & test_rc(rng) & test_rch(rng)
println(allok ? "ZOO TEST OK" : "ZOO TEST FAILED")
exit(allok ? 0 : 1)
