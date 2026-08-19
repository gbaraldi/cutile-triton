# TMA descriptor path: matmul 4096³ tf32 on H100.
using TileTriton
using TileTriton.TritonEmitter, TileTriton.TritonRun
using CUDA, Statistics
import cuTile as ct

const Spec2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
TA2(T) = ct.TileArray{T, 2, Int32, Spec2}

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

function bench(f; warmup=3, iters=50)
    for _ in 1:warmup; f(); end; CUDA.synchronize()
    minimum([CUDA.@elapsed f() for _ in 1:iters])
end

for (tm, tn, tk, warps, stages, use_tma) in [
    (128, 128, 32, 8, 3, false),   # baseline: pointer+mask
    (128, 128, 32, 8, 3, true),
    (128, 128, 64, 8, 3, true),
    (128, 128, 64, 4, 3, true),
    (256, 128, 32, 8, 3, true),
    (128, 256, 32, 8, 3, true),
]
    try
        k = TritonRun.triton_kernel(matmul, Tuple{TA2(Float32), TA2(Float32), TA2(Float32),
                                                  ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                                  ct.Constant{Int, tk}};
                                    name="matmul", num_warps=warps, num_stages=stages, use_tma=use_tma)
        grid = (cld(M, tm), cld(N, tn))
        C .= 0
        t = bench(() -> TritonRun.launch!(k, grid, A, B, C))
        ok = isapprox(Array(C), Array(A) * Array(B); rtol=1e-2) ? "" : "  WRONG RESULTS"
        println("tma=$use_tma tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: ",
                round(tflops(t); digits=1), " TFLOP/s (shared=$(k.shared)B, scratch=$(k.global_scratch_size)B)", ok)
    catch e
        println("tma=$use_tma tm=$tm tn=$tn tk=$tk warps=$warps stages=$stages: FAILED ($(first(sprint(showerror, e), 100)))")
    end
end

# Python TMA reference (BM=128 BN=128 BK=64 warps=8 stages=3), transpose trick.
let
    md = read("matmul_tma_py.json", String)
    mod = CuModule(read("matmul_tma_py.cubin"))
    fun = CuFunction(mod, "matmul_tma")
    shared = parse(Int, match(r"\"shared\":\s*(\d+)", md)[1])
    gss = parse(Int, match(r"\"global_scratch_size\":\s*(\d+)", md)[1])
    nw = parse(Int, match(r"\"num_warps\":\s*(\d+)", md)[1])
    CUDA.attributes(fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shared
    grid = (cld(N, 128), cld(M, 128))
    scratch = CuArray{UInt8}(undef, prod(grid) * gss)
    tt = Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},Int32,Int32,Int32,
               CuPtr{Cvoid},CuPtr{Cvoid}}
    C .= 0
    f = () -> cudacall(fun, tt, pointer(B), pointer(A), pointer(C),
                       Int32(N), Int32(M), Int32(K),
                       reinterpret(CuPtr{Cvoid}, pointer(scratch)), CU_NULL;
                       threads=nw * 32, blocks=grid, shmem=shared)
    t = bench(f)
    ok = isapprox(Array(C), Array(A) * Array(B); rtol=1e-2) ? "" : "  WRONG RESULTS"
    println("python-tma  BM=128 BN=128 BK=64 warps=8 stages=3: ",
            round(tflops(t); digits=1), " TFLOP/s", ok)
end
println("(reference: tileiras 171, cuBLAS 285, pointer+mask plateau ~120)")