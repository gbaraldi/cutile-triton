# Quick MI300A perf probe: streaming bandwidth + matmul TFLOPS through the
# Triton hip backend. Not a tuned benchmark — a sanity number.
using AMDGPU
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

# --- vadd bandwidth ----------------------------------------------------------
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ct.store(c; index=bid, tile=ct.load(a; index=bid, shape=(tile,)) +
                                 ct.load(b; index=bid, shape=(tile,)))
    return
end

let n = 1 << 28, tile = 2048   # 3 GiB traffic
    k = TritonRun.triton_kernel(vadd, Tuple{TA1(Float32), TA1(Float32), TA1(Float32),
                                            ct.Constant{Int, tile}}; name="vadd", num_warps=4)
    a = AMDGPU.rand(Float32, n); b = AMDGPU.rand(Float32, n); c = AMDGPU.zeros(Float32, n)
    t = timeit(() -> TritonRun.launch!(k, cld(n, tile), a, b, c))
    println("vadd     n=2^28        ", round(3 * n * 4 / t / 1e9; digits=0), " GB/s")
end

# --- matmul TFLOPS -----------------------------------------------------------
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

for (T, label) in ((Float16, "f16"), (Float32, "f32"))
    M = N = K = 4096
    tm, tn, tk = 128, 128, 64
    k = TritonRun.triton_kernel(matmul, Tuple{TA2(T), TA2(T), TA2(T),
                                              ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                              ct.Constant{Int, tk}};
                                name="matmul_$label", num_warps=8)
    A = AMDGPU.rand(T, M, K); B = AMDGPU.rand(T, K, N); C = AMDGPU.zeros(T, M, N)
    t = timeit(() -> TritonRun.launch!(k, (cld(M, tm), cld(N, tn)), A, B, C))
    println("matmul   $label 4096^3     ", round(2M * N * K / t / 1e12; digits=1), " TFLOPS")
end
println("ROCM QUICK BENCH DONE")
