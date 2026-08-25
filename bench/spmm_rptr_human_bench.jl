# Minimal harness: launch the hand-written kernel in bench/spmm_rptr_human.jl
# through TileTriton (cuTile → TTIR → Triton) on the exported flow matrix
# and report per-launch time and throughput.
#
#   julia --project=. bench/spmm_rptr_human_bench.jl [n...]   (default 8 64 256)
#   SPMM_DATA=flow_TX|vision_rnd_05; override tiles with SPMM_TILES="TM,TN,TK;..."
#   and the warp count with TRITON_NUM_WARPS (default 8, the sweep winner).
#
# The format is the node-order range+CSR (contiguous_ptr = collapsed rptr,
# scattered_* = the in-CSR) with explicit ±rsign value arrays; C is checked
# against A*B in node order. Throughput is quoted as GFLOP/s (2·nnz·n / t)
# and as GB/s of the compulsory traffic (A arrays + B + C once).

using LinearAlgebra, SparseArrays, Random, Printf, Serialization
using CUDA
import cuTile as ct
using cuTile: Adapt
using .Adapt: adapt
using TileTriton
get!(ENV, "TRITON_NUM_WARPS", "8")
TileTriton.install_shim!()          # reroutes ct.launch to Triton

include(joinpath(@__DIR__, "spmm_formats.jl"))
include(joinpath(@__DIR__, "spmm_rptr_human.jl"))

timeit(f; warmup=3, nruns=20) = (foreach(_ -> f(), 1:warmup); CUDA.synchronize();
                                 minimum(Float64(CUDA.@elapsed f()) for _ in 1:nruns))

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = Float32
    data = get(ENV, "SPMM_DATA", "flow_TX")
    # winners of the job-8138 sweep (cuTile rc vals, rptr node order): TILE_K=1,
    # 8 warps; 64×8 at n=8, 32×32 beyond
    tiles_for(n) = haskey(ENV, "SPMM_TILES") ?
        [Tuple(parse.(Int, split(s, ','))) for s in split(ENV["SPMM_TILES"], ';')] :
        n <= 8 ? [(64, 8, 1), (32, 8, 1)] : [(32, 32, 1), (16, 32, 2)]

    datadir = joinpath(@__DIR__, "data")
    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$data.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "$(data)_t.jls")))
    m, k = size(A)
    println("# spmm_rptr_human bench: triton ($(CUDA.name(CUDA.device()))), $T, $data: ",
            "A = $m×$k with nnz=$(nnz(A)), n = $ns, $(ENV["TRITON_NUM_WARPS"]) warps")

    rc = range_csr(At.colptr, At.rowval, At.nzval)
    no = node_order_rptr(rc, k)
    rsign = T(rc.rsign)
    # pad the pointer arrays so rows past m (last tile) read in bounds
    pad(v) = CuArray(Int32[v; 0])
    Ad = SplitRangeCSRMatrix(pad(no.rptr), CuArray(fill(rsign, k)),
                             pad(no.inptr), CuArray(no.inids),
                             CuArray(fill(-rsign, length(no.inids))))
    abytes = sizeof(no.rptr) + sizeof(no.inptr) + 2sizeof(no.inids) + k * sizeof(T)

    rng = MersenneTwister(42)
    α, β = T(1), T(0)
    for n in ns, (tm, tn, tk) in tiles_for(n)
        label = "human rc[$tm×$tn×$tk]"
        case = "rptr_human $data n=$n $T"
        Bh = rand(rng, T, k, n)
        Cref = (A * Bh)[no.perm, :]
        B = CuArray(Bh)
        C = CUDA.fill(T(NaN), m, n)
        bytes = abytes + (k + m) * n * sizeof(T)
        grid = (cld(m, tm), cld(n, tn))
        launch() = ct.launch(spmm_rc_kernel, grid, C, Ad, B, α, β,
                             ct.Constant(tm), ct.Constant(tn), ct.Constant(tk),
                             ct.Constant(false))
        try
            launch(); CUDA.synchronize()
            got = Array(C)
            if !isapprox(got, α .* Cref; rtol=sqrt(eps(T)) * 100)
                println("CHECK FAILED\t$label (max abs err ", maximum(abs.(got .- Cref)), ")")
                continue
            end
            t = timeit(launch)
            @printf("RESULT\t%s\t%s\t%.1f µs\t%.1f GFLOP/s\t%.1f GB/s\n",
                    case, label, t * 1e6, 2.0 * nnz(A) * n / t / 1e9, bytes / t / 1e9)
        catch err
            rethrow(err)
            println("RESULT\t$case\t$label\tFAILED\t",
                    replace(first(sprint(showerror, err), 300), "\n" => " "))
        end
        flush(stdout)
    end
    println("SPMM RPTR HUMAN BENCH DONE")
end

main()
