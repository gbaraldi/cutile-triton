# rocBLAS's tuned f16 path: gemm_ex with f32 compute (hgemm, what gemm!
# uses, is the untuned legacy path). Direct call into the generated wrapper.
using AMDGPU
using AMDGPU.rocBLAS
using AMDGPU.rocBLAS: rocblas_gemm_ex_64, rocblas_operation_none,
                      rocblas_datatype_f16_r, rocblas_datatype_f32_r,
                      rocblas_gemm_algo_standard

function timeit(f; warmup=3, nruns=20)
    for _ in 1:warmup; f(); end
    AMDGPU.synchronize()
    minimum([(AMDGPU.synchronize(); t0 = time_ns(); f(); AMDGPU.synchronize();
              (time_ns() - t0) / 1e9) for _ in 1:nruns])
end

M = N = K = 4096
A = AMDGPU.rand(Float16, M, K); B = AMDGPU.rand(Float16, K, N)
C = AMDGPU.zeros(Float16, M, N)
alpha = Ref{Float32}(1f0); beta = Ref{Float32}(0f0)

gemm_ex!() = GC.@preserve alpha beta A B C rocblas_gemm_ex_64(
    rocBLAS.handle(), rocblas_operation_none, rocblas_operation_none,
    M, N, K,
    Base.unsafe_convert(Ptr{Cvoid}, alpha),
    pointer(A), rocblas_datatype_f16_r, M,
    pointer(B), rocblas_datatype_f16_r, K,
    Base.unsafe_convert(Ptr{Cvoid}, beta),
    pointer(C), rocblas_datatype_f16_r, M,
    pointer(C), rocblas_datatype_f16_r, M,
    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, Int32(0), UInt32(0))

gemm_ex!(); AMDGPU.synchronize()
ref = Float32.(Array(A)) * Float32.(Array(B))
got = Float32.(Array(C))
relerr = maximum(abs.(got .- ref)) / maximum(abs.(ref))
println("rel err vs f32 host gemm: ", round(relerr; sigdigits=2))
relerr < 1e-2 || error("gemm_ex produced wrong results")

t = timeit(gemm_ex!)
println("RESULT\tgemm_ex f16(f32acc) 4096^3\t", round(2M * N * K / t / 1e12; digits=1), " TFLOPS")
println("DONE")
