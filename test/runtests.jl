# Correctness tests: cuTile kernels through the Triton backend vs CPU references.
# TILETRITON_TEST_BACKEND=cuda (default) | rocm picks the vendor.
using TileTriton
using TileTriton.TritonEmitter, TileTriton.TritonRun
import cuTile as ct

const BACKEND = get(ENV, "TILETRITON_TEST_BACKEND", "cuda")
@static if get(ENV, "TILETRITON_TEST_BACKEND", "cuda") == "rocm"
    using AMDGPU
    const GPU = AMDGPU
    TileTriton.use_rocm!()
else
    using CUDA
    const GPU = CUDA
end

const Spec1 = ct.ArraySpec{1}(128, true, (1,), (0,))
const Spec2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
TA1(T) = ct.TileArray{T, 1, Int32, Spec1}
TA2(T) = ct.TileArray{T, 2, Int32, Spec2}

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
    a = GPU.rand(T, n); b = GPU.rand(T, n); c = GPU.zeros(T, n)
    TritonRun.launch!(k, cld(n, tile), a, b, c)
    GPU.synchronize()
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
    a = GPU.rand(Float32, m, n)
    out = GPU.zeros(Float32, m, 1)
    TritonRun.launch!(k, cld(m, tm), a, out)
    GPU.synchronize()
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
    A = GPU.rand(Float32, M, K); B = GPU.rand(Float32, K, N); C = GPU.zeros(Float32, M, N)
    TritonRun.launch!(k, (cld(M, tm), cld(N, tn)), A, B, C)
    GPU.synchronize()
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
    x = GPU.rand(Float32, n, m)
    y = GPU.zeros(Float32, n, m)
    grid = 512  # fewer blocks than columns → exercises the persistent loop
    TritonRun.launch!(k, grid, y, x)
    GPU.synchronize()
    xh = Array(x)
    eh = exp.(xh .- maximum(xh; dims=1))
    ref = eh ./ sum(eh; dims=1)
    check("softmax $n x $m (persistent, tile=$tile)", Array(y) ≈ ref)
end

# --- strided views: cld(size, step) windows, the last one partial -----------
function copy_windows(a, b)
    src = ct.eachtile(a, (4,); step=(3,), padding_mode=ct.PaddingMode.Zero)
    dst = ct.eachtile(b, (4,); step=(3,))
    for i in 1:size(src, 1)
        dst[i] = src[i]
    end
    return
end

let n = 11   # windows start at 0, 3, 6, 9: only the partial last one covers b[11]
    k = TritonRun.triton_kernel(copy_windows, Tuple{TA1(Float32), TA1(Float32)};
                                name="copy_windows")
    a = copyto!(GPU.zeros(Float32, n), Float32.(1:n))
    b = GPU.zeros(Float32, n)
    TritonRun.launch!(k, 1, a, b)
    GPU.synchronize()
    check("strided view windows incl. partial tail", Array(b) == Array(a))
end

# --- print signedness follows the Julia element type ------------------------
function print_tile(x)
    println("v=", ct.load(x; index=Int32(1), shape=(4,)))
    return
end

for (T, signed) in ((Int32, 1), (UInt32, 0), (Bool, 0))
    io = IOBuffer()
    TritonRun.code_triton(io, print_tile, Tuple{TA1(T)}; name="print_tile")
    ttir = String(take!(io))
    check("print isSigned=$signed for $T", occursin("isSigned = array<i32: $signed>", ttir))
end

# --- TMA for 16-bit floats is only skipped before Hopper --------------------
let tt16 = Tuple{TA2(Float16), TA2(Float16)}
    target(sm) = TritonRun.TargetInfo("cuda", sm, 32, "cubin")
    check("f16 TMA kept on sm_90", TritonRun.tma_eligible(tt16, target(90)))
    check("f16 TMA skipped on sm_89", !TritonRun.tma_eligible(tt16, target(89)))
end

# --- floor/ceil division of integer tiles -----------------------------------
function round_div(qf, qc, x, d)
    a = ct.load(x; index=Int32(1), shape=(64,))
    b = ct.load(d; index=Int32(1), shape=(64,))
    ct.store(qf; index=Int32(1), tile=fld.(a, b))
    ct.store(qc; index=Int32(1), tile=cld.(a, b))
    return
end

for T in (Int32, UInt32)
    # every sign combination, exact and inexact quotients
    xs = T <: Signed ? T.(repeat(-8:7, 4)) : T.(repeat(0:15, 4) .* 5)
    ds = T <: Signed ? T.(repeat([-3, -2, 2, 3], inner=16)) : T.(repeat([1, 2, 3, 7], inner=16))
    k = TritonRun.triton_kernel(round_div, Tuple{TA1(T), TA1(T), TA1(T), TA1(T)}; name="round_div")
    qf = GPU.zeros(T, 64); qc = GPU.zeros(T, 64)
    TritonRun.launch!(k, 1, qf, qc, copyto!(GPU.zeros(T, 64), xs), copyto!(GPU.zeros(T, 64), ds))
    GPU.synchronize()
    check("fld/cld of $T tiles", Array(qf) == fld.(xs, ds) && Array(qc) == cld.(xs, ds))
end

# --- 64-bit integers: loop bounds, mixed widths, extents beyond 2^31 --------
function loop64(out, lo::Int64, hi::Int64)
    acc = zeros(Int64, (16,))
    for i in lo:hi
        acc = acc .+ i
    end
    ct.store(out; index=Int32(1), tile=acc)
    return
end

let lo = Int64(2)^32 + 5, hi = lo + 3
    k = TritonRun.triton_kernel(loop64, Tuple{TA1(Int64), Int64, Int64}; name="loop64")
    out = GPU.zeros(Int64, 16)
    TritonRun.launch!(k, 1, out, lo, hi)
    GPU.synchronize()
    check("Int64 loop bounds beyond 2^31", all(==(sum(lo:hi)), Array(out)))
end

function loop_offset(out, n::Int64, off::Int64)
    acc = zeros(Int64, (16,))
    for i in 1:n
        acc = acc .+ (i + off)
    end
    ct.store(out; index=Int32(1), tile=acc)
    return
end

let n = Int64(4), off = Int64(2)^33
    k = TritonRun.triton_kernel(loop_offset, Tuple{TA1(Int64), Int64, Int64}; name="loop_offset")
    out = GPU.zeros(Int64, 16)
    TritonRun.launch!(k, 1, out, n, off)
    GPU.synchronize()
    check("Int64 induction variable with Int64 operands", all(==(sum(i + off for i in 1:n)), Array(out)))
end

# ~2 GiB each: skipped when the device cannot hold them
function try_alloc(T, dims...)
    try
        return GPU.zeros(T, dims...)
    catch err
        println("  skip: cannot allocate $(Base.format_bytes(sizeof(T) * prod(dims))) ($(typeof(err)))")
        return nothing
    end
end

function count_tiles(out, a)
    ct.store(out; index=Int32(1), tile=zeros(Int64, (16,)) .+ Int64(ct.num_tiles(a, 1, (1024,))))
    return
end

let len = Int64(2)^31 + 8192   # a single dimension beyond Int32: Int64-indexed
    a = try_alloc(Int8, len)
    if a !== nothing
        TAi64 = ct.TileArray{Int8, 1, Int64, Spec1}
        k = TritonRun.triton_kernel(count_tiles, Tuple{TA1(Int64), TAi64}; name="count_tiles")
        out = GPU.zeros(Int64, 16)
        TritonRun.launch!(k, 1, out, a)
        GPU.synchronize()
        check("tile count of a dimension beyond 2^31", Array(out)[1] == cld(len, 1024))
    end
    a = nothing; GC.gc()
end

function last_tile(out, a, j::Int32)
    ct.store(out; index=(Int32(1), Int32(1)), tile=ct.load(a; index=(Int32(1), j), shape=(16, 64)))
    return
end

let ncol = 2^27 + 64   # Int32-indexed, but 16 × ncol = 2^31 + 1024 elements
    a = try_alloc(Int8, 16, ncol)
    if a !== nothing
        fill!(view(a, :, ncol-63:ncol), Int8(7))
        # pointer path: descriptors address in hardware at 64 bits
        k = TritonRun.triton_kernel(last_tile, Tuple{TA2(Int8), TA2(Int8), Int32};
                                    name="last_tile", use_tma=false)
        out = GPU.zeros(Int8, 16, 64)
        TritonRun.launch!(k, 1, out, a, Int32(cld(ncol, 64)))
        GPU.synchronize()
        check("pointer offsets beyond 2^31 elements (Int32-indexed)", all(==(Int8(7)), Array(out)))
        check("small in-tile span keeps the i32-offset build", k.wide[] === nothing)
    end
    a = nothing; GC.gc()
end

function first_tile(out, a)
    ct.store(out; index=(Int32(1), Int32(1)), tile=ct.load(a; index=(Int32(1), Int32(1)), shape=(16, 64)))
    return
end

# Int32-indexed and tall-skinny: one 16 × 64 tile spans 63 × m > 2^31 elements
let m = 2^25 + 2^20, n = 64
    a = try_alloc(Int8, m, n)
    if a !== nothing
        a[1:16, n] .= Int8(7)
        k = TritonRun.triton_kernel(first_tile, Tuple{TA2(Int8), TA2(Int8)};
                                    name="first_tile", use_tma=false)
        out = GPU.zeros(Int8, 16, 64)
        TritonRun.launch!(k, 1, out, GPU.zeros(Int8, 16, 64))
        check("in-tile span within i32 launches the i32-offset build", k.wide[] === nothing)
        TritonRun.launch!(k, 1, out, a)
        GPU.synchronize()
        h = Array(out)
        check("in-tile span beyond 2^31 launches the i64-offset build",
              k.wide[] !== nothing && all(==(Int8(7)), h[:, n]) && all(iszero, h[:, 1:n-1]))
    end
    a = nothing; GC.gc()
end

function last_window(out, a, j::Int32)
    src = ct.eachtile(a, (4,); step=(3,), padding_mode=ct.PaddingMode.Zero)
    ct.store(out; index=Int32(1), tile=src[j])
    return
end

# length typemax(Int32), windows of 4 every 3: the last one starts at len - 1,
# so its bounds check counts past 2^31 (the bytes beyond `len` hold a sentinel)
let len = Int(typemax(Int32))
    buf = try_alloc(Int8, len + 16)
    if buf !== nothing
        fill!(buf, Int8(9)); buf[len:len] .= Int8(7)
        k = TritonRun.triton_kernel(last_window, Tuple{TA1(Int8), TA1(Int8), Int32}; name="last_window")
        out = GPU.fill(Int8(-1), 4)
        TritonRun.launch!(k, 1, out, view(buf, 1:len), Int32(cld(len, 3)))
        GPU.synchronize()
        check("bounds check of a window ending past 2^31", Array(out) == Int8[7, 0, 0, 0])
    end
    buf = nothing; GC.gc()
end

# --- candidate racing applies a read-modify-write kernel exactly once -------
function accumulate!(acc, x, tile::Int)
    bid = ct.bid(1)
    a = ct.load(acc; index=bid, shape=(tile,))
    ct.store(acc; index=bid, tile=a + ct.load(x; index=bid, shape=(tile,)))
    return
end

TileTriton.install_shim!()
let n = 4096, tile = 256
    acc = GPU.ones(Float32, n)
    x = GPU.ones(Float32, n)
    ct.launch(accumulate!, cld(n, tile), acc, x, ct.Constant(tile))   # races candidates
    GPU.synchronize()
    check("racing restores arguments (one application)", all(==(2f0), Array(acc)))
end

# --- a failing ct.@assert aborts with its message ---------------------------
# It takes the context down, hence a subprocess. cuTile's own version of this
# test runs its native backend in the child, so it covers neither ROCm nor us.
let script = """
    using TileTriton
    import cuTile as ct
    @static if get(ENV, "TILETRITON_TEST_BACKEND", "cuda") == "rocm"
        using AMDGPU; const GPU = AMDGPU; TileTriton.use_rocm!()
    else
        using CUDA; const GPU = CUDA
    end
    TileTriton.install_shim!()
    function assert_fail(a, tile::Int)
        bid = ct.bid(1)
        ct.@assert bid > Int32(999999) "custom assert message"
        ct.store(a; index=bid, tile=ct.load(a; index=bid, shape=(tile,)))
        return
    end
    a = GPU.ones(Float32, 1024)
    ct.launch(assert_fail, 8, a, ct.Constant(128))
    GPU.synchronize()
    """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
    output = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout=output, stderr=output); wait=false)
    close(output.in)
    reader = @async read(output, String)
    wait(proc)
    check("failing ct.@assert aborts with its message",
          proc.exitcode != 0 && contains(fetch(reader), "custom assert message"))
end

println("ALL $npass TESTS PASSED")
