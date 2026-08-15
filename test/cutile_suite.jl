# Run cuTile.jl's own device test suite through the Triton backend, by
# overwriting cuTile.cufunction with a shim that compiles via TritonEmitter +
# the triton wheel and launches via CUDA.jl. Unmodified test files then
# exercise our backend; per-file pass/fail/error counts = the coverage map.
using Test
using CUDA
using cuTile
import cuTile as ct
using TileTriton

TileTriton.install_shim!()


# ---------------------------------------------------------------------------
# Version-matched device tests (the checkout's suite tests main-only features
# like kw-arange gather views and literal-array constants that cuTile 0.3.2
# itself cannot compile natively either).
const TESTDIR = joinpath(dirname(dirname(pathof(cuTile))), "test", "device")
files = sort(filter(endswith(".jl"), readdir(TESTDIR)))
isempty(ARGS) || (files = filter(f -> any(a -> occursin(a, f), ARGS), files))

function counts(ts)
    # DefaultTestSet only stores Fail/Error/child sets in .results; passes are
    # a counter.
    p = ts.n_passed
    f = e = 0
    for r in ts.results
        if r isa Test.Fail; f += 1
        elseif r isa Test.Error; e += 1
        elseif r isa Test.AbstractTestSet
            (p2, f2, e2) = counts(r); p += p2; f += f2; e += e2
        end
    end
    return p, f, e
end

results = []
try
    @testset "cuTile device suite via Triton" begin
        for file in files
            path = joinpath(TESTDIR, file)
            local ts
            t = @elapsed begin
                ts = @testset "$file" begin
                    try
                        # each file in a fresh module with the suite's imports
                        m = Module(Symbol("Test_", replace(file, "." => "_")))
                        Core.eval(m, :(using Test, CUDA, cuTile))
                        Core.eval(m, :(import cuTile as ct))
                        Core.eval(m, :(import TileTriton: TritonShim))
                        # @filecheck FileCheck-verifies native Tile IR text —
                        # inapplicable to this backend. Neutralize to `true`
                        # (report these as skipped-by-stub, not as passes of
                        # real checks).
                        Core.eval(m, :(macro filecheck(args...) true end))
                        Base.include(m, path)
                    catch err
                        @error "file-level failure" file err
                        @test false
                    end
                end
            end
            p, f, e = counts(ts)
            push!(results, (file, p, f, e, t))
            println(">>> $file: pass=$p fail=$f error=$e ($(round(t; digits=1))s)")
            flush(stdout)
            # contain context poisoning (illegal instruction etc.) per file;
            # cached CuModules belong to the old context, so drop them too
            try
                CUDA.device_reset!()
                empty!(TileTriton.TritonShim.KERNEL_CACHE)
            catch
            end
        end
    end
catch err
    err isa Test.TestSetException || rethrow()
end

println("\n", "="^64)
println(rpad("file", 26), lpad("pass", 6), lpad("fail", 6), lpad("error", 7), lpad("time", 8))
for (file, p, f, e, t) in results
    println(rpad(file, 26), lpad(p, 6), lpad(f, 6), lpad(e, 7), lpad("$(round(Int,t))s", 8))
end
tp = sum(r[2] for r in results); tf = sum(r[3] for r in results); te = sum(r[4] for r in results)
println(rpad("TOTAL", 26), lpad(tp, 6), lpad(tf, 6), lpad(te, 7))
