# Benchmark: cuTile kernels via (a) native tileiras, (b) Julia→TTIR→triton,
# (c) Python→triton (same tile configs), plus CUDA.jl broadcast / cuBLAS.
using TileTriton
using TileTriton.TritonEmitter, TileTriton.TritonRun
using CUDA
using LinearAlgebra
using Statistics
import cuTile as ct

const Spec1 = typeof(ct.ArraySpec{1}(128, true, (1,), (0,)))
const Spec2 = typeof(ct.ArraySpec{2}(128, true, (1, 0), (0, 0)))
TA1(T) = ct.TileArray{T, 1, Spec1}
TA2(T) = ct.TileArray{T, 2, Spec2}

function timeit(f; warmup=3, iters=50)
    for _ in 1:warmup; f(); end
    CUDA.synchronize()
    ts = [CUDA.@elapsed f() for _ in 1:iters]
    return minimum(ts), median(ts)
end

# Load a Python-compiled reference kernel.
struct PyKernel
    fun::CuFunction
    mod::CuModule
    num_warps::Int
    warp_size::Int
    shared::Int
end
function load_py_kernel(name)
    dir = joinpath(@__DIR__, "bench_py")
    md = read(joinpath(dir, name * ".json"), String)
    mod = CuModule(read(joinpath(dir, name * ".cubin")))
    kname = match(r"\"name\":\s*\"(\w+)\"", md)[1]
    fun = CuFunction(mod, String(kname))
    nw = parse(Int, match(r"\"num_warps\":\s*(\d+)", md)[1])
    ws = parse(Int, match(r"\"warp_size\":\s*(\d+)", md)[1])
    sh = parse(Int, match(r"\"shared\":\s*(\d+)", md)[1])
    if sh > 48 * 1024
        CUDA.attributes(fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = sh
    end
    return PyKernel(fun, mod, nw, ws, sh)
end

results = Dict{String,Dict{String,Float64}}()
row!(bench, who, val) = (get!(results, bench, Dict{String,Float64}())[who] = val)

# ============================================================================
# vadd: n = 2^27 Float32, tile 1024. Metric: GB/s (2 reads + 1 write).
# ============================================================================
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ta = ct.load(a; index=bid, shape=(tile,))
    tb = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=ta + tb)
    return
end

let n = 2^27, tile = 1024, T = Float32
    gbs(t) = 3 * n * sizeof(T) / t / 1e9
    a = CUDA.rand(T, n); b = CUDA.rand(T, n); c = CUDA.zeros(T, n)
    grid = cld(n, tile)

    kj = TritonRun.triton_kernel(vadd, Tuple{TA1(T), TA1(T), TA1(T), ct.Constant{Int, tile}};
                                 name="vadd", num_warps=4)
    t, _ = timeit(() -> TritonRun.launch!(kj, grid, a, b, c))
    row!("vadd", "julia-triton", gbs(t))

    kp = load_py_kernel("vadd_py")
    tt_args = Tuple{CuPtr{T},CuPtr{T},CuPtr{T},Int32,CuPtr{Cvoid},CuPtr{Cvoid}}
    t, _ = timeit(() -> cudacall(kp.fun, tt_args, pointer(a), pointer(b), pointer(c),
                                 Int32(n), CU_NULL, CU_NULL;
                                 threads=kp.num_warps * kp.warp_size, blocks=grid, shmem=kp.shared))
    row!("vadd", "python-triton", gbs(t))

    t, _ = timeit(() -> ct.launch(vadd, grid, a, b, c, ct.Constant(tile)))
    row!("vadd", "cuTile-native", gbs(t))

    t, _ = timeit(() -> (c .= a .+ b))
    row!("vadd", "CUDA.jl bcast", gbs(t))

    @assert Array(c) ≈ Array(a) .+ Array(b)
end

# ============================================================================
# matmul: 4096^3 Float32 (TF32 tensor cores), tiles 128x128x32. TFLOP/s.
# ============================================================================
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

let M = 4096, N = 4096, K = 4096, tm = 128, tn = 128, tk = 32
    tflops(t) = 2.0 * M * N * K / t / 1e12
    A = CUDA.rand(Float32, M, K); B = CUDA.rand(Float32, K, N); C = CUDA.zeros(Float32, M, N)
    grid = (cld(M, tm), cld(N, tn))

    kj = TritonRun.triton_kernel(matmul, Tuple{TA2(Float32), TA2(Float32), TA2(Float32),
                                               ct.Constant{Int, tm}, ct.Constant{Int, tn},
                                               ct.Constant{Int, tk}};
                                 name="matmul", num_warps=8)
    t, _ = timeit(() -> TritonRun.launch!(kj, grid, A, B, C))
    row!("matmul", "julia-triton", tflops(t))
    @assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-2)

    # The Python tutorial kernel is written row-major; feed it the transpose
    # identity C^T = B^T * A^T so its inner axis is the contiguous one, exactly
    # like the Julia-emitted kernel's layout convention.
    kp = load_py_kernel("matmul_py")
    ttm = Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},
                Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,
                CuPtr{Cvoid},CuPtr{Cvoid}}
    grid_py = (cld(N, tm), cld(M, tn))
    t, _ = timeit(() -> cudacall(kp.fun, ttm, pointer(B), pointer(A), pointer(C),
                                 Int32(N), Int32(M), Int32(K),
                                 Int32(stride(B, 2)), Int32(stride(B, 1)),
                                 Int32(stride(A, 2)), Int32(stride(A, 1)),
                                 Int32(stride(C, 2)), Int32(stride(C, 1)),
                                 CU_NULL, CU_NULL;
                                 threads=kp.num_warps * kp.warp_size, blocks=grid_py, shmem=kp.shared))
    row!("matmul", "python-triton", tflops(t))
    @assert isapprox(Array(C), Array(A) * Array(B); rtol=1e-2)

    t, _ = timeit(() -> ct.launch(matmul, grid, A, B, C,
                                  ct.Constant(tm), ct.Constant(tn), ct.Constant(tk)))
    row!("matmul", "cuTile-native", tflops(t))

    CUDA.math_mode!(CUDA.FAST_MATH)  # allow TF32 in cuBLAS, like the kernels
    t, _ = timeit(() -> mul!(C, A, B))
    row!("matmul", "cuBLAS tf32", tflops(t))
end

# ============================================================================
# softmax: 1000 rows x 32768 cols, tile 1024, persistent grid. GB/s (r+w).
# ============================================================================
function softmax(output, input, TILE_SIZE::Int)
    ct.@compiler_options occupancy=2
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

let n = 1000, m = 32768, tile = 1024, grid = 1024
    gbs(t) = 2 * n * m * sizeof(Float32) / t / 1e9
    x = CUDA.rand(Float32, n, m); y = CUDA.zeros(Float32, n, m)

    kj = TritonRun.triton_kernel(softmax, Tuple{TA2(Float32), TA2(Float32),
                                                ct.Constant{Int, tile}};
                                 name="softmax", num_warps=4)
    t, _ = timeit(() -> TritonRun.launch!(kj, grid, y, x))
    row!("softmax", "julia-triton", gbs(t))

    kp = load_py_kernel("softmax_py")
    tts = Tuple{CuPtr{Float32},CuPtr{Float32},Int32,Int32,Int32,Int32,
                CuPtr{Cvoid},CuPtr{Cvoid}}
    t, _ = timeit(() -> cudacall(kp.fun, tts, pointer(y), pointer(x),
                                 Int32(n), Int32(m), Int32(stride(y, 2)), Int32(stride(x, 2)),
                                 CU_NULL, CU_NULL;
                                 threads=kp.num_warps * kp.warp_size, blocks=grid, shmem=kp.shared))
    row!("softmax", "python-triton", gbs(t))

    t, _ = timeit(() -> ct.launch(softmax, grid, y, x, ct.Constant(tile)))
    row!("softmax", "cuTile-native", gbs(t))

    xh = Array(x); eh = exp.(xh .- maximum(xh; dims=1))
    @assert Array(y) ≈ eh ./ sum(eh; dims=1)
end

# ============================================================================
println("\n== Results (H100 NVL, min of 50 runs) ==")
for bench in ("vadd", "matmul", "softmax")
    unit = bench == "matmul" ? "TFLOP/s" : "GB/s"
    println("\n--- $bench ($unit) ---")
    for (who, val) in sort(collect(results[bench]); by=x -> -x[2])
        println(rpad(who, 16), round(val; digits=1))
    end
end
