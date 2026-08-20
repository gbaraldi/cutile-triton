# The bench SpMM tile kernels (CSR rows, JDS pm, 2-per-row pm) on the flow
# matrix through cuTile's native Tile IR backend (tileiras) vs TileTriton,
# from the same kernel source. Kernels go through `cuTile.launch`, which is
# what TileTriton's shim reroutes (it replaces `cuTile.cufunction`).
#
#   CUTILE_BACKEND=native|triton julia --project=. bench/spmm_backends.jl [n...]
#
# Prints RESULT rows in the bench/spmm_flow.jl format with the backend in
# the implementation label, so the two logs concatenate into one table via
# bench/spmm_flow_tables.py. Needs bench/data/flow_TX{,_t,_zoo}.jls and a
# tileiras that supports the device (CUDA ≥ 13.2 for sm_89).

using LinearAlgebra, SparseArrays, Random, Serialization
using CUDA
import cuTile as ct
using TileTriton

const BACKEND = get(ENV, "CUTILE_BACKEND", "native")
BACKEND in ("native", "triton") || error("CUTILE_BACKEND must be native or triton")
BACKEND == "triton" && TileTriton.install_shim!()

include(joinpath(@__DIR__, "spmm_csr_kernels.jl"))
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    return minimum(Float64(CUDA.@elapsed f()) for _ in 1:nruns)
end

function check(name, Cd, Cref; rtol)
    ok = isapprox(Cd, Cref; rtol)
    ok || println("CHECK FAILED\t$name (max abs err ", maximum(abs.(Cd .- Cref)), ")")
    return ok
end

result_row(case, impl, t, flops) =
    println("RESULT\t$case\t$impl\t$(round(t * 1e6; digits=1)) µs\t",
            round(flops / t / 1e9; digits=1), " GFLOP/s")

fail_row(case, impl, err) =
    println("RESULT\t$case\t$impl\tFAILED\t",
            replace(first(sprint(showerror, err), 100), "\n" => " "))

"Verify (against the device-resident reference) and time one launch closure."
function bench_launch(case, label, flops, Cd, Cref, rtol, launch)
    try
        fill!(Cd, NaN32)
        launch()
        CUDA.synchronize()
        check(label, Cd, Cref; rtol) || return
        result_row(case, label, timeit(launch), flops)
    catch err
        fail_row(case, label, err)
    end
    flush(stdout)
end

# small per-n config sets (the winners of the spmm_flow.jl sweeps)
csr_cfgs(n) = n <= 8 ? [(8, 8, 16), (16, 8, 4)] : [(8, 64, 8), (8, 64, 4)]
zoo_cfgs(n) = n <= 8 ? [(32, 8)] : [(32, 64), (32, 16)]

function main()
    ns = isempty(ARGS) ? [8, 64, 256] : parse.(Int, ARGS)
    T = Float32
    datadir = joinpath(@__DIR__, "data")
    A = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "flow_TX.jls")))
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "flow_TX_t.jls")))
    zoo = deserialize(joinpath(datadir, "flow_TX_zoo.jls"))
    m, k = size(A)
    println("# spmm_backends bench: $BACKEND ($(CUDA.name(CUDA.device()))), $T, ",
            "A = $m×$k with nnz=$(nnz(A)), n = $ns")
    BACKEND == "native" && ct.versioninfo()
    rng = MersenneTwister(42)
    rtol = sqrt(eps(T)) * 100
    be = BACKEND

    csrA = (rowptr=CuArray(At.colptr), colval=CuArray(At.rowval), nzval=CuArray(At.nzval))
    csrAt = (rowptr=CuArray(A.colptr), colval=CuArray(A.rowval), nzval=CuArray(A.nzval))
    jds_col = CuArray(zoo.jds_colidx)
    jds_iter = CuArray(zoo.jds_iterptr)
    ndiag = Int32(length(zoo.jds_iterptr) - 1)
    tpr_col2 = CuArray(zoo.tpr_colidx)

    for (mat, csr, Amat) in (("A", csrA, A), ("At", csrAt, At))
        mm, kk = size(Amat)
        for n in ns
            case = "flow $mat n=$n $T"
            flops = 2.0 * nnz(Amat) * n
            Bh = rand(rng, T, kk, n)
            Bd = CuArray(Bh)
            Cref = CuArray(Amat * Bh)
            Cd = similar(Cref)
            one_, zero_ = T(1), T(0)

            for (tm, tn, tk) in csr_cfgs(n)
                label = "$be csr[$(tm)×$(tn)×$(tk)]"
                bench_launch(case, label, flops, Cd, Cref, rtol, () ->
                    ct.launch(spmm_csr_rows_kernel, spmm_grid(Cd, tm, tn),
                              Cd, csr.rowptr, csr.colval, csr.nzval, Bd, one_, zero_,
                              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk),
                              ct.Constant(false)))
            end
            for (tm, tn) in zoo_cfgs(n)
                if mat == "A"
                    label = "$be jds pm[$(tm)×$(tn)]"
                    bench_launch(case, label, flops, Cd, Cref, rtol, () ->
                        ct.launch(spmm_jds_pm_kernel, spmm_grid(Cd, tm, tn),
                                  Cd, jds_col, jds_iter, Bd, one_, zero_, ndiag,
                                  ct.Constant(tm), ct.Constant(tn), ct.Constant(false),
                                  ct.Constant(false)))
                else
                    label = "$be 2pr pm[$(tm)×$(tn)]"
                    bench_launch(case, label, flops, Cd, Cref, rtol, () ->
                        ct.launch(spmm_2pr_pm_kernel, spmm_grid(Cd, tm, tn),
                                  Cd, tpr_col2, Bd, one_, zero_,
                                  ct.Constant(tm), ct.Constant(tn), ct.Constant(false),
                                  ct.Constant(false)))
                end
            end
            Bd = Cd = Cref = nothing
            GC.gc(); CUDA.reclaim()
        end
    end
    println("SPMM BACKENDS BENCH DONE ($BACKEND)")
end

main()
