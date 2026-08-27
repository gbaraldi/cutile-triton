# Minimal harness: launch the hand-written kernel in bench/spmm_rptr_human.jl
# through TileTriton (cuTile → TTIR → Triton) on the exported flow matrix
# and report per-launch time and throughput.
#
#   julia --project=. bench/spmm_rptr_human_bench.jl [n...]   (default 8 64 256)
#   SPMM_DATA=flow_TX|vision_rnd_05; override tiles with SPMM_TILES="TM,TN,TK;..."
#   and the warp count with TRITON_NUM_WARPS (default 8, the sweep winner).
#   SPMM_INT=Int64 switches the index type (default Int32).
#   SPMM_HEAVY_MAXLEN=<len> peels rows with more nonzeros into the hybrid's
#   dense per-row path (bench/spmm_rptr_hybrid_human.jl); default 64 for the
#   vision instances (two ~92k-entry rows), off (0) otherwise.
#
# The format is the node-order range+CSR (contiguous_lo/hi = the collapsed
# rptr split into per-row [lo, hi) bounds, scattered_* = the in-CSR) with
# explicit ±rsign value arrays; C is checked against A*B in node order. Throughput is quoted as GFLOP/s (2·nnz·n / t)
# and as GB/s of the compulsory traffic (A arrays + B + C once).

using LinearAlgebra, SparseArrays, Random, Printf, Serialization
using CUDA
import cuTile as ct
using cuTile: Adapt
using .Adapt: adapt
# SPMM_TRITON=0 runs the kernel on native cuTile (tileiras) instead of the
# TileTriton shim, for a same-backend comparison with the zoo kernels.
const USE_TRITON = get(ENV, "SPMM_TRITON", "1") != "0"
if USE_TRITON
    using TileTriton
    get!(ENV, "TRITON_NUM_WARPS", "8")
    TileTriton.install_shim!()      # reroutes ct.launch to Triton
end

include(joinpath(@__DIR__, "spmm_formats.jl"))
include(joinpath(@__DIR__, "spmm_rptr_human.jl"))
include(joinpath(@__DIR__, "spmm_rptr_hybrid_human.jl"))

timeit(f; warmup=3, nruns=20) = (foreach(_ -> f(), 1:warmup); CUDA.synchronize();
                                 minimum(Float64(CUDA.@elapsed f()) for _ in 1:nruns))

# Ti is the index type of every host/device index array (SPMM_INT=Int64 to
# override); spmm_formats.jl derives its result types from its inputs.
function main(Ti::Type{<:Integer} = getfield(Base, Symbol(get(ENV, "SPMM_INT", "Int32"))))
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = Float32
    data = get(ENV, "SPMM_DATA", "flow_TX")
    # winners of the job-8138 sweep (cuTile rc vals, rptr node order): TILE_K=1,
    # 8 warps; 64×8 at n=8, 32×32 beyond
    tiles_for(n) = haskey(ENV, "SPMM_TILES") ?
        [Tuple(parse.(Int, split(s, ','))) for s in split(ENV["SPMM_TILES"], ';')] :
        n <= 8 ? [(64, 8, 1), (32, 8, 1)] : [(32, 32, 1), (16, 32, 2)]

    datadir = joinpath(@__DIR__, "data")
    A = SparseMatrixCSC{T, Ti}(deserialize(joinpath(datadir, "$data.jls")))
    At = SparseMatrixCSC{T, Ti}(deserialize(joinpath(datadir, "$(data)_t.jls")))
    m, k = size(A)
    backend = USE_TRITON ? "triton, $(ENV["TRITON_NUM_WARPS"]) warps" : "native cuTile"
    println("# spmm_rptr_human bench: $backend ($(CUDA.name(CUDA.device()))), $T/$Ti, $data: ",
            "A = $m×$k with nnz=$(nnz(A)), n = $ns")

    rc = range_csr(At.colptr, At.rowval, At.nzval)
    no = node_order_rptr(rc, k)
    rsign = T(rc.rsign)
    # node-order rc arrays (lo, exclusive hi; the kernel zero-pads its row
    # loads, so nothing needs padding). Heavy rows are emptied here and
    # handled densely by the hybrid; that must happen after the permutation
    # since an emptied range would break node_order_rptr's chain check.
    heavy_maxlen = parse(Int, get(ENV, "SPMM_HEAVY_MAXLEN",
                                  startswith(data, "vision") ? "64" : "0"))
    rcn = (; lo=no.lo, hi=no.hi, inptr=no.inptr, inids=no.inids, rsign=rc.rsign)
    light, heavy = heavy_maxlen > 0 ? split_heavy(rcn; maxlen=heavy_maxlen) :
                                      (rcn, (; row=Ti[], len=Ti[]))
    nice = SplitRangeCSRMatrix(CuArray(light.lo), CuArray(light.hi), CuArray(fill(rsign, k)),
                               CuArray(light.inptr), CuArray(light.inids),
                               CuArray(fill(-rsign, length(light.inids))))
    abytes = sizeof(light.lo) + sizeof(light.hi) + sizeof(light.inptr) +
             2sizeof(light.inids) + k * sizeof(T)
    Ad = if isempty(heavy.row)
        nice
    else
        println("# hybrid: $(length(heavy.row)) heavy rows (> $heavy_maxlen nnz), ",
                "$(sum(heavy.len)) of $(nnz(A)) nnz handled densely")
        odd = Dict{Ti, Tuple{CuVector{Ti}, CuMatrix{T}}}()
        for (h, i) in enumerate(heavy.row)
            ids = Ti[heavy.lo[h]:(heavy.hi[h] - 1); heavy.ids[heavy.ptr[h]:(heavy.ptr[h + 1] - 1)]]
            vals = T[fill(rsign, heavy.hi[h] - heavy.lo[h]); fill(-rsign, heavy.ptr[h + 1] - heavy.ptr[h])]
            odd[i] = (CuArray(ids), CuArray(reshape(vals, :, 1)))
            abytes += sizeof(ids) + sizeof(vals)
        end
        HybridSparseMatrix(nice, odd)
    end

    rng = MersenneTwister(42)
    α, β = T(1), T(0)
    for n in ns, (tm, tn, tk) in tiles_for(n)
        label = (Ad isa HybridSparseMatrix ? "human rch" : "human rc") * "[$tm×$tn×$tk]"
        case = "rptr_human $(USE_TRITON ? "triton" : "cutile") $data n=$n $T"
        Bh = rand(rng, T, k, n)
        Cref = (A * Bh)[no.perm, :]
        B = CuArray(Bh)
        C = CUDA.fill(T(NaN), m, n)
        bytes = abytes + (k + m) * n * sizeof(T)
        launch() = Ad isa HybridSparseMatrix ?
            spmm_hybrid!(C, Ad, B, α, β, tm, tn, tk, false) :
            launch_rc!(C, Ad, B, α, β, tm, tn, tk, false)
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
