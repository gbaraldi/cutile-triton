# Benchmark all cuTile examples on one backend, printing machine-readable rows.
#   CUTILE_BACKEND=native|triton julia --project=. bench_examples.jl [names...]
using CUDA
using cuTile
import cuTile as ct
using Statistics

const BACKEND = get(ENV, "CUTILE_BACKEND", "native")
if BACKEND == "triton"
    using TileTriton
    TileTriton.install_shim!()
end

const EXDIR = joinpath(dirname(dirname(pathof(cuTile))), "examples")
const NRUNS = 20
const WARMUP = 5

names = isempty(ARGS) ? sort([replace(f, ".jl" => "") for f in readdir(EXDIR)
                              if endswith(f, ".jl") && f != "benchmarks.jl"]) : ARGS

for name in names
    file = joinpath(EXDIR, name * ".jl")
    isfile(file) || continue
    try
        mod = Module(Symbol("Bench_", name))
        Base.include(mod, file)
        (isdefined(mod, :prepare) && isdefined(mod, :run)) || continue
        data = @invokelatest mod.prepare(; benchmark=true)
        metric_result = isdefined(mod, :metric) ? (@invokelatest mod.metric(data)) : nothing
        result = @invokelatest mod.run(data; nruns=NRUNS, warmup=WARMUP)
        results = if hasproperty(result, :times)
            t = result.times
            t isa Dict ? t : Dict{String,Vector{Float64}}("cuTile" => t)
        elseif hasproperty(result, :times_fwd)
            Dict("cuTile Fwd" => result.times_fwd, "cuTile Bwd" => result.times_bwd)
        else
            continue
        end
        # baselines (GPUArrays/SIMT/cuBLAS/FFTW...) only on the native pass
        if BACKEND == "native" && isdefined(mod, :run_others)
            merge!(results, @invokelatest mod.run_others(data; nruns=NRUNS, warmup=WARMUP))
        end
        for (impl, times) in results
            isempty(times) && continue
            tmin = minimum(times)
            m = metric_result isa Dict ? get(metric_result, impl, nothing) : metric_result
            metr = m === nothing ? "" :
                   m[2] == "GB/s" ? string(round(Int, m[1] / (tmin / 1e3) / 1e9), " GB/s") :
                   m[2] == "TFLOPS" ? string(round(m[1] / (tmin / 1e3) / 1e12; digits=1), " TFLOPS") :
                   string(round(Int, tmin * 1000), " µs")
            println("RESULT\t$name\t$impl\t$(round(tmin; digits=4))\t$(round(mean(times); digits=4))\t$metr")
        end
        flush(stdout)
    catch err
        msg = replace(first(sprint(showerror, err), 120), "\n" => " ")
        println("RESULT\t$name\tFAILED\t\t\t$msg")
        flush(stdout)
    end
end
println("BENCH DONE ($BACKEND)")
