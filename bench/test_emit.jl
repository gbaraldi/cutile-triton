# Emit TTIR for the three coverage kernels and write .ttir files.
using TileTriton.TritonEmitter
import cuTile as ct

function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ta = ct.load(a; index=bid, shape=(tile,))
    tb = ct.load(b; index=bid, shape=(tile,))
    ct.store(c; index=bid, tile=ta + tb)
    return
end

function rowsum(a, out, tm::Int, tn::Int)
    bid = ct.bid(1)
    t = ct.load(a; index=(bid, Int32(1)), shape=(tm, tn))
    s = sum(t; dims=2)
    ct.store(out; index=(bid, Int32(1)), tile=s)
    return
end

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

const Spec1 = typeof(ct.ArraySpec{1}(128, true, (1,), (0,)))
const Spec2 = typeof(ct.ArraySpec{2}(128, true, (1, 0), (0, 0)))
TA1(T) = ct.TileArray{T, 1, Spec1}
TA2(T) = ct.TileArray{T, 2, Spec2}

for (name, f, argtypes) in [
    ("vadd", vadd, Tuple{TA1(Float32), TA1(Float32), TA1(Float32), ct.Constant{Int, 1024}}),
    ("rowsum", rowsum, Tuple{TA2(Float32), TA2(Float32), ct.Constant{Int, 64}, ct.Constant{Int, 128}}),
    ("matmul", matmul, Tuple{TA2(Float32), TA2(Float32), TA2(Float32),
                             ct.Constant{Int, 128}, ct.Constant{Int, 128}, ct.Constant{Int, 32}}),
]
    print("emitting $name ... ")
    text, spec = emit_ttir(f, argtypes; name=name)
    write(joinpath(@__DIR__, "$(name)_emitted.ttir"), text)
    println("ok ($(length(text)) chars, $(length(spec)) args)")
end
println("EMIT OK")
