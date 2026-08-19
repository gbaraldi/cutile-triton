# Dump post-pass StructuredIRCode for the kernels the TTIR emitter must cover.
using cuTile
import cuTile as ct

# --- vadd 1D (view-based load/store) ---
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ta = ct.load(a; index=bid, shape=(tile,))
    tb = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=ta + tb)
    return
end

# --- row sum: 2D -> 1D reduction ---
function rowsum(a, out, tm::Int, tn::Int)
    bid = ct.bid(1)
    t = ct.load(a; index=(bid, Int32(1)), shape=(tm, tn))
    s = sum(t; dims=2)
    ct.store(out; index=(bid, Int32(1)), tile=s)
    return
end

# --- matmul (K loop, carries, mma) ---
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

const Spec1 = ct.ArraySpec{1}(128, true, (1,), (0,))
const Spec2 = ct.ArraySpec{2}(128, true, (1, 0), (0, 0))
const TA1 = ct.TileArray{Float32, 1, Int32, Spec1}
const TA2 = ct.TileArray{Float32, 2, Int32, Spec2}

function dump(name, f, argtypes)
    println("="^70)
    println("== ", name)
    println("="^70)
    for (sci, ret) in ct.code_structured(f, argtypes; optimize=true)
        show(stdout, MIME"text/plain"(), sci)
        println("\n-- argtypes:")
        for (i, T) in enumerate(sci.argtypes)
            println("  arg ", i, ": ", T)
        end
    end
end

dump("vadd", vadd, Tuple{TA1, TA1, TA1, ct.Constant{Int, 1024}})
dump("rowsum", rowsum, Tuple{TA2, TA2, ct.Constant{Int, 64}, ct.Constant{Int, 128}})
dump("matmul", matmul, Tuple{TA2, TA2, TA2, ct.Constant{Int, 128}, ct.Constant{Int, 128}, ct.Constant{Int, 32}})
