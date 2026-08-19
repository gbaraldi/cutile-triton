# fp16 matmul: does the descriptor/TMA path pay off for f16 where tf32 didn't?
using TileTriton
using TileTriton.TritonEmitter, TileTriton.TritonRun
using CUDA, LinearAlgebra, Statistics
import cuTile as ct

const Spec2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
TA2(T) = ct.TileArray{T, 2, Int32, Spec2}

function matmul16(A, B, C, tm::Int, tn::Int, tk::Int)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    num_k = ct.num_tiles(A, 2, (tm, tk))
    acc = zeros(Float32, tm, tn)
    for k in Int32(1):num_k
        a = ct.load(A; index=(bid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.Zero)
        b = ct.load(B; index=(k, bid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.Zero)
        acc = muladd(a, b, acc)
    end
    ct.store(C; index=(bid_m, bid_n), tile=convert(ct.Tile{Float16}, acc))
    return nothing
end

M = N = K = 4096
A = CUDA.rand(Float16, M, K); B = CUDA.rand(Float16, K, N); C = CUDA.zeros(Float16, M, N)
tflops(t) = 2.0 * M * N * K / t / 1e12
ref = Float32.(Array(A)) * Float32.(Array(B))

function bench(f; warmup=3, iters=50)
    for _ in 1:warmup; f(); end; CUDA.synchronize()
    minimum([CUDA.@elapsed f() for _ in 1:iters])
end

for (tm, tn, tk, warps, stages, use_tma) in [
    (128, 128, 64, 8, 3, false),
    (128, 128, 64, 8, 3, true),
    (128, 256, 64, 8, 3, true),
    (128, 256, 64, 8, 4, true),
    (256, 128, 64, 8, 3, true),
]
    try
        k = TritonRun.triton_kernel(matmul16, Tuple{TA2(Float16), TA2(Float16), TA2(Float16),
                                                    ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                                    ct.Constant{Int, tk}};
                                    name="matmul16", num_warps=warps, num_stages=stages, use_tma=use_tma)
        grid = (cld(M, tm), cld(N, tn))
        C .= 0
        t = bench(() -> TritonRun.launch!(k, grid, A, B, C))
        err = maximum(abs.(Float32.(Array(C)) .- ref)) / maximum(abs.(ref))
        ok = err < 2e-2 ? "" : "  WRONG (rel err $err)"
        println("tma=$use_tma tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: ",
                round(tflops(t); digits=1), " TFLOP/s (shared=$(k.shared)B)", ok)
    catch e
        println("tma=$use_tma tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: FAILED ($(first(sprint(showerror, e), 100)))")
    end
end

# native cuTile (tileiras) same kernel
let tm = 128, tn = 128, tk = 64
    grid = (cld(M, tm), cld(N, tn))
    C .= 0
    t = bench(() -> ct.launch(matmul16, grid, A, B, C,
                              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))
    err = maximum(abs.(Float32.(Array(C)) .- ref)) / maximum(abs.(ref))
    println("cuTile-native tm=$tm tn=$tn tk=$tk: ", round(tflops(t); digits=1),
            " TFLOP/s", err < 2e-2 ? "" : "  WRONG")
end

# cuBLAS f16
let
    CUDA.math_mode!(CUDA.FAST_MATH)
    t = bench(() -> mul!(C, A, B))
    println("cuBLAS f16: ", round(tflops(t); digits=1), " TFLOP/s")
end