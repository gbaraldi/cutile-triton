# Micro-bench isolating where the cuTile 2-per-row pm kernel loses to the
# one-thread-per-element KA kernel on Aᵀ·B of the flow matrix. Variants of
# spmm_2pr_pm_kernel differing in how the id tiles and the B slabs are
# fetched and how C is written; TTIR op counts are dumped per variant so
# reshapes / layout changes can be correlated with the timings.
#
#   TRITON_KERNEL_INFO=1 julia --project=. bench/spmm_2pr_variants.jl [n...]

using LinearAlgebra, SparseArrays, Random, Serialization
using CUDA
import cuTile as ct
using TileTriton
using TileTriton: TritonRun

include(joinpath(@__DIR__, "spmm_common.jl"))
include(joinpath(@__DIR__, "spmm_zoo_kernels.jl"))

const TTIR_DIR = mktempdir()
ENV["TRITON_DUMP_TTIR"] = TTIR_DIR

# --- variants ----------------------------------------------------------------

# V1: ids by flat 1-D gather (stride-2 indices into vec(colidx)), B by the
# 2-D bounds-checked gather, C by the eachtile block store.
function v1_flatids(C::ct.TileArray{T, 2}, colidx::ct.TileArray{Int32, 1},
                    B::ct.TileArray{T, 2}, alpha::T, beta::T,
                    TILE_M::Int, TILE_N::Int) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    j1 = ct.gather(colidx, Int32(2) .* rows .- Int32(1))
    j2 = ct.gather(colidx, Int32(2) .* rows)
    ncols = reshape(tile_span(bn, TILE_N), (1, TILE_N))
    b1 = ct.gather(B, (reshape(j1, (TILE_M, 1)), ncols))
    b2 = ct.gather(B, (reshape(j2, (TILE_M, 1)), ncols))
    epilogue_store(C, bm, bn, b1 .- b2, alpha, beta, TILE_M, TILE_N, false)
    return
end

# V2: V1 ids, B by a flat 1-D gather with precomputed column offsets and one
# explicit mask (no per-element 2-D bounds check), block store.
function v2_flatb(C::ct.TileArray{T, 2}, colidx::ct.TileArray{Int32, 1},
                  Bf::ct.TileArray{T, 1}, alpha::T, beta::T, k::Int32, n::Int32,
                  TILE_M::Int, TILE_N::Int) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    j1 = reshape(ct.gather(colidx, Int32(2) .* rows .- Int32(1)), (TILE_M, 1))
    j2 = reshape(ct.gather(colidx, Int32(2) .* rows), (TILE_M, 1))
    cols = tile_span(bn, TILE_N)
    coff = reshape((cols .- Int32(1)) .* k, (1, TILE_N))
    cmask = reshape(cols .<= n, (1, TILE_N))
    b1 = ct.gather(Bf, j1 .+ coff; mask=(j1 .> Int32(0)) .& cmask, check_bounds=false)
    b2 = ct.gather(Bf, j2 .+ coff; mask=(j2 .> Int32(0)) .& cmask, check_bounds=false)
    epilogue_store(C, bm, bn, b1 .- b2, alpha, beta, TILE_M, TILE_N, false)
    return
end

# V3: V2 fetches, C by scatter instead of the block store.
function v3_scatter(C::ct.TileArray{T, 2}, colidx::ct.TileArray{Int32, 1},
                    Bf::ct.TileArray{T, 1}, alpha::T, beta::T, k::Int32, n::Int32,
                    TILE_M::Int, TILE_N::Int) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    rows = tile_span(bm, TILE_M)
    j1 = reshape(ct.gather(colidx, Int32(2) .* rows .- Int32(1)), (TILE_M, 1))
    j2 = reshape(ct.gather(colidx, Int32(2) .* rows), (TILE_M, 1))
    cols = tile_span(bn, TILE_N)
    coff = reshape((cols .- Int32(1)) .* k, (1, TILE_N))
    cmask = reshape(cols .<= n, (1, TILE_N))
    b1 = ct.gather(Bf, j1 .+ coff; mask=(j1 .> Int32(0)) .& cmask, check_bounds=false)
    b2 = ct.gather(Bf, j2 .+ coff; mask=(j2 .> Int32(0)) .& cmask, check_bounds=false)
    cidx = (reshape(rows, (TILE_M, 1)), reshape(cols, (1, TILE_N)))
    ct.scatter(C, cidx, alpha .* (b1 .- b2))
    return
end

# V4: current kernel's eachtile ids, but flat-B fetch (isolates the id path).
function v4_eachtile_flatb(C::ct.TileArray{T, 2}, colidx::ct.TileArray{Int32, 2},
                           Bf::ct.TileArray{T, 1}, alpha::T, beta::T, k::Int32, n::Int32,
                           TILE_M::Int, TILE_N::Int) where {T}
    bm = ct.bid(1)
    bn = ct.bid(2)
    slots = ct.eachtile(colidx, (1, TILE_M); padding_mode=ct.PaddingMode.Zero)
    j1 = reshape(ct.load(slots, (Int32(1), bm)), (TILE_M, 1))
    j2 = reshape(ct.load(slots, (Int32(2), bm)), (TILE_M, 1))
    cols = tile_span(bn, TILE_N)
    coff = reshape((cols .- Int32(1)) .* k, (1, TILE_N))
    cmask = reshape(cols .<= n, (1, TILE_N))
    b1 = ct.gather(Bf, j1 .+ coff; mask=(j1 .> Int32(0)) .& cmask, check_bounds=false)
    b2 = ct.gather(Bf, j2 .+ coff; mask=(j2 .> Int32(0)) .& cmask, check_bounds=false)
    epilogue_store(C, bm, bn, b1 .- b2, alpha, beta, TILE_M, TILE_N, false)
    return
end

# KA reference: one thread per element, β = 0
@kernel function ka_2pr_pm!(c, colidx, b)
    i, kk = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(c))
        j = colidx[1, i]; j > 0 && (s += b[j, kk])
        j = colidx[2, i]; j > 0 && (s -= b[j, kk])
        c[i, kk] = s
    end
end

# --- harness ---------------------------------------------------------------

function ttir_stats(name)
    files = filter(f -> startswith(f, name * "_w"), readdir(TTIR_DIR))
    isempty(files) && return ""
    src = read(joinpath(TTIR_DIR, files[end]), String)
    cnt(op) = count(op, src)
    return "reshape=$(cnt("tt.reshape")) expand=$(cnt("tt.expand_dims")) " *
           "bcast=$(cnt("tt.broadcast")) load=$(cnt("tt.load")) store=$(cnt("tt.store")) " *
           "muli=$(cnt("arith.muli")) cmpi=$(cnt("arith.cmpi"))"
end

function main()
    ns = isempty(ARGS) ? [64, 256] : parse.(Int, ARGS)
    T = Float32
    datadir = joinpath(@__DIR__, "data")
    At = SparseMatrixCSC{T, Int32}(deserialize(joinpath(datadir, "flow_TX_t.jls")))
    zoo = deserialize(joinpath(datadir, "flow_TX_zoo.jls"))
    m, k = size(At)
    println("# 2pr variants: $(device_name()), Aᵀ = $m×$k nnz=$(nnz(At)), n = $ns")
    rng = MersenneTwister(42)
    rtol = sqrt(eps(T)) * 100
    col2 = CuArray(zoo.tpr_colidx)
    col1 = vec(col2)

    for n in ns
        case = "flow At n=$n $T"
        flops = 2.0 * nnz(At) * n
        Bh = rand(rng, T, k, n)
        Bd = CuArray(Bh)
        Bf = vec(Bd)
        Cref = CuArray(At * Bh)
        Cd = similar(Cref)
        ctx = (; T, Cref, Cref2=Cref, C0=Cref, rtol)
        ta2 = spmm_ta2(T); ta1 = spmm_ta1(T); ti1 = spmm_ta1(Int32); ti2 = spmm_ta2(Int32)

        # SHAPES="tm,tn,nw;..." restricts the sweep (V0 only) to the given configs
        shapes = haskey(ENV, "SHAPES") ?
            [Tuple(parse.(Int, split(s, ","))) for s in split(ENV["SHAPES"], ";")] :
            [(32, 16, 8), (32, 8, 8), (32, 64, 4), (64, 16, 8)]
        for (tm, tn, nw) in shapes
            cfg = "$(tm)×$(tn)×w$(nw)"
            consts = (ct.Constant{Int, tm}, ct.Constant{Int, tn})
            variants = [
                ("V0 current", () -> build_spmm_2pr(T; tile_m=tm, tile_n=tn, pm=true,
                                                    beta_nz=false, num_warps=nw),
                 (f!, C) -> f!(C, col2, Bd, 1, 0), "spmm_2pr_pm"),
                ("V1 flat ids", () -> begin
                    kk = TritonRun.triton_kernel(v1_flatids,
                        Tuple{ta2, ti1, ta2, T, T, consts...}; name="v1_flatids", num_warps=nw)
                    (C, a...) -> TritonRun.launch!(kk, spmm_grid(C, tm, tn), C, col1, Bd, T(1), T(0))
                 end, (f!, C) -> f!(C), "v1_flatids"),
                ("V2 flat ids+B", () -> begin
                    kk = TritonRun.triton_kernel(v2_flatb,
                        Tuple{ta2, ti1, ta1, T, T, Int32, Int32, consts...}; name="v2_flatb", num_warps=nw)
                    (C, a...) -> TritonRun.launch!(kk, spmm_grid(C, tm, tn), C, col1, Bf, T(1), T(0), Int32(k), Int32(n))
                 end, (f!, C) -> f!(C), "v2_flatb"),
                ("V3 flat+scatter", () -> begin
                    kk = TritonRun.triton_kernel(v3_scatter,
                        Tuple{ta2, ti1, ta1, T, T, Int32, Int32, consts...}; name="v3_scatter", num_warps=nw)
                    (C, a...) -> TritonRun.launch!(kk, spmm_grid(C, tm, tn), C, col1, Bf, T(1), T(0), Int32(k), Int32(n))
                 end, (f!, C) -> f!(C), "v3_scatter"),
                ("V4 eachtile ids+flat B", () -> begin
                    kk = TritonRun.triton_kernel(v4_eachtile_flatb,
                        Tuple{ta2, ti2, ta1, T, T, Int32, Int32, consts...}; name="v4_eachtile_flatb", num_warps=nw)
                    (C, a...) -> TritonRun.launch!(kk, spmm_grid(C, tm, tn), C, col2, Bf, T(1), T(0), Int32(k), Int32(n))
                 end, (f!, C) -> f!(C), "v4_eachtile_flatb"),
            ]
            haskey(ENV, "SHAPES") && (variants = variants[1:1])
            for (label, build, launch, kname) in variants
                f! = try
                    build()
                catch err
                    fail_row(case, "$label[$cfg]", err); continue
                end
                t = timeit_checked(case, "$label[$cfg]", flops, Cd, ctx, C -> launch(f!, C))
                t === nothing || println("TTIR\t$label[$cfg]\t", ttir_stats(kname))
            end
        end
        ka! = ka_2pr_pm!(KA.get_backend(Cd))
        timeit_checked(case, "KA 2pr pm", flops, Cd, ctx, C -> ka!(C, col2, Bd; ndrange=(m, n)))
        Bd = Bf = Cd = Cref = nothing
        GC.gc(); CUDA.reclaim()
    end
    println("VARIANTS DONE")
end

function timeit_checked(case, label, flops, Cd, ctx, f0)
    fill!(Cd, NaN32)
    f0(Cd)
    device_sync()
    check(label, Cd, ctx.Cref; rtol=ctx.rtol) || return nothing
    t = timeit(() -> f0(Cd))
    result_row(case, label, t, flops)
    flush(stdout)
    return t
end

main()
