# Head-to-head on ROCm: the Triton backend vs native AMDGPU.jl for the same
# workloads — a hand-written @roc SIMT kernel / GPUArrays / rocBLAS, whichever
# is the idiomatic native baseline. Prints RESULT rows like bench/examples.jl.
using AMDGPU
using LinearAlgebra
using TileTriton
using TileTriton.TritonRun
import cuTile as ct

TileTriton.use_rocm!()

const Spec1 = ct.ArraySpec{1}(128, true, (1,), (0,))
const Spec2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
TA1(T) = ct.TileArray{T, 1, Int32, Spec1}
TA2(T) = ct.TileArray{T, 2, Int32, Spec2}

function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup; f(); end
    AMDGPU.synchronize()
    minimum([(AMDGPU.synchronize(); t0 = time_ns(); f(); AMDGPU.synchronize();
              (time_ns() - t0) / 1e9) for _ in 1:nruns])
end

row(name, impl, t, metr) = println("RESULT\t$name\t$impl\t$(round(t * 1e6; digits=1)) µs\t$metr")

# === vadd: 3 GiB of traffic =================================================
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ct.store(c; index=bid, tile=ct.load(a; index=bid, shape=(tile,)) +
                                 ct.load(b; index=bid, shape=(tile,)))
    return
end

# grid-stride so the launch config can't affect correctness
function vadd_roc!(c, a, b)
    i = workitemIdx().x + (workgroupIdx().x - Int32(1)) * workgroupDim().x
    stride = gridItemDim().x
    while i <= length(c)
        @inbounds c[i] = a[i] + b[i]
        i += stride
    end
    return
end

let n = 1 << 28, tile = 2048
    a = AMDGPU.rand(Float32, n); b = AMDGPU.rand(Float32, n); c = AMDGPU.zeros(Float32, n)
    gbs(t) = string(round(Int, 3 * n * 4 / t / 1e9), " GB/s")

    k = TritonRun.triton_kernel(vadd, Tuple{TA1(Float32), TA1(Float32), TA1(Float32),
                                            ct.Constant{Int, tile}}; name="vadd", num_warps=4)
    TritonRun.launch!(k, cld(n, tile), a, b, c); AMDGPU.synchronize()
    @assert Array(c[1:1000]) ≈ Array(a[1:1000]) .+ Array(b[1:1000])
    t = timeit(() -> TritonRun.launch!(k, cld(n, tile), a, b, c))
    row("vadd", "triton", t, gbs(t))

    fill!(c, 0)
    groups = cld(n, 256)
    @roc groupsize=256 gridsize=groups vadd_roc!(c, a, b); AMDGPU.synchronize()
    @assert Array(c[1:1000]) ≈ Array(a[1:1000]) .+ Array(b[1:1000])
    t = timeit(() -> @roc groupsize=256 gridsize=groups vadd_roc!(c, a, b))
    row("vadd", "AMDGPU @roc", t, gbs(t))

    t = timeit(() -> (c .= a .+ b))
    row("vadd", "broadcast", t, gbs(t))
end

# === rowsum: reduction along dim 2 ==========================================
function rowsum(a, out, tm::Int, tn::Int)
    bid = ct.bid(1)
    t = ct.load(a; index=(bid, Int32(1)), shape=(tm, tn))
    ct.store(out; index=(bid, Int32(1)), tile=sum(t; dims=2))
    return
end

let m = 131072, n = 512, tm = 32, tn = 512
    a = AMDGPU.rand(Float32, m, n)
    out = AMDGPU.zeros(Float32, m, 1)
    gbs(t) = string(round(Int, m * n * 4 / t / 1e9), " GB/s")

    k = TritonRun.triton_kernel(rowsum, Tuple{TA2(Float32), TA2(Float32),
                                              ct.Constant{Int, tm}, ct.Constant{Int, tn}};
                                name="rowsum", num_warps=4)
    TritonRun.launch!(k, cld(m, tm), a, out); AMDGPU.synchronize()
    @assert Array(out) ≈ sum(Array(a); dims=2)
    t = timeit(() -> TritonRun.launch!(k, cld(m, tm), a, out))
    row("rowsum", "triton", t, gbs(t))

    t = timeit(() -> sum!(out, a))
    row("rowsum", "GPUArrays sum!", t, gbs(t))
end

# === softmax over columns ===================================================
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

let n = 4096, m = 16384, tile = 4096
    x = AMDGPU.rand(Float32, n, m); y = AMDGPU.zeros(Float32, n, m)
    tmp = AMDGPU.zeros(Float32, 1, m)
    gbs(t) = string(round(Int, 2 * n * m * 4 / t / 1e9), " GB/s")

    k = TritonRun.triton_kernel(softmax, Tuple{TA2(Float32), TA2(Float32),
                                               ct.Constant{Int, tile}};
                                name="softmax", num_warps=4)
    grid = 1024
    TritonRun.launch!(k, grid, y, x); AMDGPU.synchronize()
    xh = Array(x); eh = exp.(xh .- maximum(xh; dims=1))
    @assert Array(y) ≈ eh ./ sum(eh; dims=1)
    t = timeit(() -> TritonRun.launch!(k, grid, y, x))
    row("softmax", "triton", t, gbs(t))

    native! = () -> begin
        maximum!(tmp, x)
        y .= exp.(x .- tmp)
        sum!(tmp, y)
        y ./= tmp
    end
    native!()
    @assert Array(y) ≈ eh ./ sum(eh; dims=1)
    t = timeit(native!)
    row("softmax", "GPUArrays", t, gbs(t))
end

# === matmul vs rocBLAS ======================================================
function matmul(A, B, C, tm::Int, tn::Int, tk::Int)
    bid_m = ct.bid(1); bid_n = ct.bid(2)
    num_k = ct.num_tiles(A, 2, (tm, tk))
    acc = zeros(Float32, tm, tn)
    for k in Int32(1):num_k
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=convert(ct.Tile{eltype(C)}, acc))
    return nothing
end

for (T, label) in ((Float16, "matmul-f16"), (Float32, "matmul-f32"))
    M = N = K = 4096
    tm, tn, tk = 128, 128, 64
    A = AMDGPU.rand(T, M, K); B = AMDGPU.rand(T, K, N); C = AMDGPU.zeros(T, M, N)
    tflops(t) = string(round(2M * N * K / t / 1e12; digits=1), " TFLOPS")

    k = TritonRun.triton_kernel(matmul, Tuple{TA2(T), TA2(T), TA2(T),
                                              ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                              ct.Constant{Int, tk}};
                                name="matmul", num_warps=8)
    t = timeit(() -> TritonRun.launch!(k, (cld(M, tm), cld(N, tn)), A, B, C))
    row(label, "triton", t, tflops(t))

    try
        t = timeit(() -> mul!(C, A, B))
        row(label, "rocBLAS", t, tflops(t))
    catch err
        println("RESULT\t$label\trocBLAS\tFAILED\t", first(sprint(showerror, err), 80))
    end
end
println("ROCM COMPARE DONE")
