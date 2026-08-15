# Step 3: load the triton-compiled cubin with CUDA.jl and launch it.
# ABI (from the PTX): x_ptr, y_ptr, out_ptr, n, global_scratch, profile_scratch.
using CUDA

md_text = read(joinpath(@__DIR__, "vadd_julia.json"), String)
num_warps = parse(Int, match(r"\"num_warps\":\s*(\d+)", md_text)[1])
warp_size = parse(Int, match(r"\"warp_size\":\s*(\d+)", md_text)[1])
shared = parse(Int, match(r"\"shared\":\s*(\d+)", md_text)[1])
name = String(match(r"\"name\":\s*\"(\w+)\"", md_text)[1])

mod = CuModule(read(joinpath(@__DIR__, "vadd_julia.cubin")))
f = CuFunction(mod, name)

n = 1000
BLOCK = 128
x = CUDA.rand(Float32, n)
y = CUDA.rand(Float32, n)
out = CUDA.zeros(Float32, n)

cudacall(
    f,
    Tuple{CuPtr{Float32},CuPtr{Float32},CuPtr{Float32},Int32,CuPtr{Cvoid},CuPtr{Cvoid}},
    pointer(x), pointer(y), pointer(out), Int32(n), CU_NULL, CU_NULL;
    threads=num_warps * warp_size, blocks=cld(n, BLOCK), shmem=shared,
)
synchronize()

@assert Array(out) ≈ Array(x) .+ Array(y)
println("PASS: Julia-emitted TTIR -> triton wheel -> cubin -> CUDA.jl launch, results correct")
