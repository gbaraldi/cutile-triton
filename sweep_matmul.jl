# Quick tile-config sweep for the Julia→Triton matmul on H100.
include(joinpath(@__DIR__, "TritonEmitter.jl"))
include(joinpath(@__DIR__, "TritonRun.jl"))
using .TritonEmitter, .TritonRun
using CUDA, Statistics
import cuTile as ct

const Spec2 = typeof(ct.ArraySpec{2}(128, true, (1, 0), (0, 0)))
TA2(T) = ct.TileArray{T, 2, Spec2}

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

M = N = K = 4096
A = CUDA.rand(Float32, M, K); B = CUDA.rand(Float32, K, N); C = CUDA.zeros(Float32, M, N)
tflops(t) = 2.0 * M * N * K / t / 1e12

for (tm, tn, tk, warps, stages) in [
    (128, 128, 32, 8, 4), (128, 128, 32, 8, 5), (128, 128, 32, 8, 6),
    (128, 128, 32, 4, 4), (64, 128, 32, 4, 4), (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 3),


]
    try
        k = TritonRun.triton_kernel(matmul, Tuple{TA2(Float32), TA2(Float32), TA2(Float32),
                                                  ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                                  ct.Constant{Int, tk}};
                                    name="matmul", num_warps=warps, num_stages=stages)
        grid = (cld(M, tm), cld(N, tn))
        f = () -> TritonRun.launch!(k, grid, A, B, C)
        for _ in 1:3; f(); end; CUDA.synchronize()
        t = minimum([CUDA.@elapsed f() for _ in 1:30])
        println("tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: ",
                round(tflops(t); digits=1), " TFLOP/s")
    catch e
        println("tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: FAILED ($(first(sprint(showerror, e), 120)))")
    end
end
@assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-2)
println("sweep done")