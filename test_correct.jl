# Correctness tests: cuTile kernels through the Triton backend vs CPU references.
include(joinpath(@__DIR__, "TritonEmitter.jl"))
include(joinpath(@__DIR__, "TritonRun.jl"))
using .TritonEmitter, .TritonRun
using CUDA
import cuTile as ct

const Spec1 = typeof(ct.ArraySpec{1}(128, true, (1,), (0,)))
const Spec2 = typeof(ct.ArraySpec{2}(128, true, (1, 0), (0, 0)))
TA1(T) = ct.TileArray{T, 1, Spec1}
TA2(T) = ct.TileArray{T, 2, Spec2}

npass = 0
function check(name, ok)
    global npass
    ok || error("FAIL: $name")
    npass += 1
    println("  pass: $name")
end

# --- vadd ------------------------------------------------------------------
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ta = ct.load(a; index=bid, shape=(tile,))
    tb = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=ta + tb)
    return
end

for T in (Float32, Float16, Float64), n in (1_024_000, 999_999)  # non-divisible too
    tile = 1024
    k = TritonRun.triton_kernel(vadd, Tuple{TA1(T), TA1(T), TA1(T), ct.Constant{Int, tile}};
                                name="vadd", num_warps=4)
    a = CUDA.rand(T, n); b = CUDA.rand(T, n); c = CUDA.zeros(T, n)
    TritonRun.launch!(k, cld(n, tile), a, b, c)
    CUDA.synchronize()
    check("vadd $T n=$n", Array(c) ≈ Array(a) .+ Array(b))
end

# --- rowsum (reduction) -----------------------------------------------------
function rowsum(a, out, tm::Int, tn::Int)
    bid = ct.bid(1)
    t = ct.load(a; index=(bid, Int32(1)), shape=(tm, tn))
    s = sum(t; dims=2)
    ct.store(out; index=(bid, Int32(1)), tile=s)
    return
end

let m = 4096, n = 128, tm = 64, tn = 128
    k = TritonRun.triton_kernel(rowsum, Tuple{TA2(Float32), TA2(Float32),
                                              ct.Constant{Int, tm}, ct.Constant{Int, tn}};
                                name="rowsum", num_warps=4)
    a = CUDA.rand(Float32, m, n)
    out = CUDA.zeros(Float32, m, 1)
    TritonRun.launch!(k, cld(m, tm), a, out)
    CUDA.synchronize()
    check("rowsum $m x $n", Array(out) ≈ sum(Array(a); dims=2))
end

# --- matmul (K loop, tf32 mma, padding) --------------------------------------
function matmul(A, B, C, tm::Int, tn::Int, tk::Int)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    num_k = ct.num_tiles(A, 2, (tm, tk))
    acc = zeros(Float32, tm, tn)
    for k in Int32(1):num_k
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        a = convert(ct.Tile{ct.TFloat32}, a)
        b = convert(ct.Tile{ct.TFloat32}, b)
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=convert(ct.Tile{Float32}, acc))
    return nothing
end

for (M, N, K) in ((512, 512, 512), (300, 260, 200))  # divisible + ragged
    tm, tn, tk = 128, 128, 32
    k = TritonRun.triton_kernel(matmul, Tuple{TA2(Float32), TA2(Float32), TA2(Float32),
                                              ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                              ct.Constant{Int, tk}};
                                name="matmul", num_warps=8)
    A = CUDA.rand(Float32, M, K); B = CUDA.rand(Float32, K, N); C = CUDA.zeros(Float32, M, N)
    TritonRun.launch!(k, (cld(M, tm), cld(N, tn)), A, B, C)
    CUDA.synchronize()
    ref = Array(A) * Array(B)
    got = Array(C)
    err = maximum(abs.(got .- ref)) / max(maximum(abs.(ref)), 1)
    check("matmul $M x $N x $K (tf32, rel err $(round(err; sigdigits=2)))", err < 1e-3)
end

# --- softmax (persistent while loop, NegInf padding, exp/max/broadcast) -----
function softmax(output, input, TILE_SIZE::Int)
    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    M = size(input, 2)
    col_idx = pid
    while col_idx <= M
        col = ct.load(input; index=(Int32(1), col_idx), shape=(TILE_SIZE, 1),
                      padding_mode=ct.PaddingMode.NegInf)
        col = convert(ct.Tile{Float32}, col)
        col_max = maximum(col; dims=1)
        numerator = exp.(col .- col_max)
        denominator = sum(numerator; dims=1)
        ct.store(output; index=(Int32(1), col_idx), tile=numerator ./ denominator)
        col_idx += num_programs
    end
    return
end

let n = 1000, m = 2048, tile = 1024   # n < tile → masked + NegInf-padded
    k = TritonRun.triton_kernel(softmax, Tuple{TA2(Float32), TA2(Float32),
                                               ct.Constant{Int, tile}};
                                name="softmax", num_warps=4)
    x = CUDA.rand(Float32, n, m)
    y = CUDA.zeros(Float32, n, m)
    grid = 512  # fewer blocks than columns → exercises the persistent loop
    TritonRun.launch!(k, grid, y, x)
    CUDA.synchronize()
    xh = Array(x)
    eh = exp.(xh .- maximum(xh; dims=1))
    ref = eh ./ sum(eh; dims=1)
    check("softmax $n x $m (persistent, tile=$tile)", Array(y) ≈ ref)
end

println("ALL $npass TESTS PASSED")
