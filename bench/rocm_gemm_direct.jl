# Direct rocBLAS gemm! numbers (AMDGPU.jl's mul! on Julia 1.12 routes Float16
# through GPUArrays' generic kernel instead of rocblas hgemm).
using AMDGPU
using AMDGPU.rocBLAS

function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup; f(); end
    AMDGPU.synchronize()
    minimum([(AMDGPU.synchronize(); t0 = time_ns(); f(); AMDGPU.synchronize();
              (time_ns() - t0) / 1e9) for _ in 1:20])
end

for T in (Float16, Float32)
    M = N = K = 4096
    A = AMDGPU.rand(T, M, K); B = AMDGPU.rand(T, K, N); C = AMDGPU.zeros(T, M, N)
    try
        t = timeit(() -> rocBLAS.gemm!('N', 'N', one(T), A, B, zero(T), C))
        println("RESULT\tgemm-direct $T\t", round(2M * N * K / t / 1e12; digits=1), " TFLOPS")
    catch err
        println("RESULT\tgemm-direct $T\tFAILED\t", first(sprint(showerror, err), 100))
    end
end
println("DONE")
