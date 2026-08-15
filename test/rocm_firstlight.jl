# First light for the ROCm target: emit → compile (hsaco) → load → launch →
# verify, in separate stages so failures localize.
using AMDGPU
using TileTriton
using TileTriton.TritonRun
import cuTile as ct

println("[1] AMDGPU functional: ", AMDGPU.functional())
dev = AMDGPU.device()
println("    device: ", dev)

println("[2] use_rocm!")
TileTriton.use_rocm!()

const S1 = typeof(ct.ArraySpec{1}(128, true, (1,), (16,)))
function vadd(a, b, c, tile::Int)
    bid = ct.bid(1)
    ct.store(c; index=bid, tile=ct.load(a; index=bid, shape=(tile,)) + ct.load(b; index=bid, shape=(tile,)))
    return
end
TA = ct.TileArray{Float32,1,S1}
tt = Tuple{TA, TA, TA, ct.Constant{Int, 1024}}

println("[3] emit")
text, = TileTriton.emit_ttir(vadd, tt; name="vadd")
println("    ttir: ", length(text), " chars")

println("[4] compile (hip backend)")
k = TritonRun.triton_kernel(vadd, tt; name="vadd", num_warps=4)
println("    ok: warp_size=", k.warp_size, " shared=", k.shared,
        " scratch=", k.global_scratch_size)

println("[5] launch")
n = 1_048_576
a = AMDGPU.rand(Float32, n); b = AMDGPU.rand(Float32, n); c = AMDGPU.zeros(Float32, n)
TritonRun.launch!(k, cld(n, 1024), a, b, c)
AMDGPU.synchronize()

println("[6] verify")
ok = Array(c) ≈ Array(a) .+ Array(b)
println(ok ? "ROCM FIRST LIGHT: PASS" : "WRONG RESULTS")
ok || exit(1)

# quick bandwidth number
f = () -> TritonRun.launch!(k, cld(n, 1024), a, b, c)
for _ in 1:3; f(); end; AMDGPU.synchronize()
ts = [(AMDGPU.synchronize(); t0 = time_ns(); f(); AMDGPU.synchronize(); (time_ns() - t0) / 1e9) for _ in 1:20]
gbs = 3 * n * 4 / minimum(ts) / 1e9
println("vadd bandwidth: ", round(gbs; digits=0), " GB/s (n=", n, ")")
