# Shared harness for the SpMM benches (spmm_csr.jl, spmm_flow.jl): backend
# selection, timing, verification, and the CoolPDLP / vendor-library
# baselines. Environment knobs:
#   SPMM_BACKEND=cuda|rocm   force the backend (default: cuda if functional,
#                            else rocm — needs AMDGPU in the environment:
#                            julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")')

const BACKEND = let req = lowercase(get(ENV, "SPMM_BACKEND", "auto"))
    if req == "auto"
        ok = try
            @eval using CUDA
            CUDA.functional()
        catch
            false
        end
        ok ? "cuda" : "rocm"
    elseif req in ("cuda", "rocm")
        req
    else
        error("SPMM_BACKEND must be cuda or rocm")
    end
end

if BACKEND == "cuda"
    using CUDA
    const GPUArr = CuArray
    device_sync() = CUDA.synchronize()
    device_name() = CUDA.name(CUDA.device())
    # @eval: the @elapsed macro can only expand once its module is loaded
    @eval device_elapsed(f) = CUDA.@elapsed f()
else
    using AMDGPU
    const GPUArr = ROCArray
    device_sync() = AMDGPU.synchronize()
    device_name() = AMDGPU.HIP.name(AMDGPU.device())
    @eval device_elapsed(f) = AMDGPU.@elapsed f()
end

using TileTriton
using TileTriton.TritonRun
import cuTile as ct
BACKEND == "rocm" && TileTriton.use_rocm!()

# KernelAbstractions is an indirect dependency here, so load it by UUID
# rather than adding it to the project.
const KA = Base.require(Base.PkgId(
    Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"), "KernelAbstractions"))
using .KA  # for @kernel/@index

# CoolPDLP baseline (src/utils/mat_csr.jl spmm_csr!): one thread per (row,
# rhs column).
@kernel function spmm_csr_ka!(c, A_rowptr, A_colval, A_nzval, b, α, β)
    i, batch_idx = @index(Global, NTuple)
    s = zero(eltype(c))
    for k in A_rowptr[i]:(A_rowptr[i + Int32(1)] - Int32(1))
        j = A_colval[k]
        s += A_nzval[k] * b[j, batch_idx]
    end
    c[i, batch_idx] = α * s + β * c[i, batch_idx]
end

function vendor_csr(rowptr, colval, nzval, m, k)
    if BACKEND == "cuda"
        return CUDA.CUSPARSE.CuSparseMatrixCSR(rowptr, colval, nzval, (m, k))
    else
        return AMDGPU.rocSPARSE.ROCSparseMatrixCSR(rowptr, colval, nzval, (m, k))
    end
end
const VENDOR_NAME = BACKEND == "cuda" ? "cuSPARSE" : "rocSPARSE"

# ---------------------------------------------------------------------------

# Timed with device events: bracketing a host clock with synchronize() rounds
# anything beyond ~2 ms up to the ~1 ms granularity of the host-side wait.
function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup
        f()
    end
    device_sync()
    best = Inf
    for _ in 1:nruns
        best = min(best, Float64(device_elapsed(f)))
    end
    return best
end

# Compares on the device when Cref lives there (no D2H transfer of C).
function check(name, Cd, Cref; rtol)
    Ccmp = Cref isa Array ? Array(Cd) : Cd
    ok = isapprox(Ccmp, Cref; rtol)
    ok || println("CHECK FAILED\t$name (max abs err ",
                  maximum(abs.(Ccmp .- Cref)), ")")
    return ok
end

result_row(case, impl, t, flops) =
    println("RESULT\t$case\t$impl\t$(round(t * 1e6; digits=1)) µs\t",
            round(flops / t / 1e9; digits=1), " GFLOP/s")

fail_row(case, impl, err) =
    println("RESULT\t$case\t$impl\tFAILED\t",
            replace(first(sprint(showerror, err), 100), "\n" => " "))
