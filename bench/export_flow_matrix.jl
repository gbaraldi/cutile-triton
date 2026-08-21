# Build a min-cost-flow constraint matrix (as in CoolPDLP2/reactant/benchmarks.jl)
# and serialize it (and its transpose) as SparseMatrixCSC{Float32,Int32}, plus
# the raw zoo formats.
#
# Run in an environment that has MinimumCostFlows (not a dep of this repo):
#   julia --project=<mcf env> bench/export_flow_matrix.jl <name> <dimacs files...>
#   e.g.  flow_TX ~/min-cost-flow/road/road_flow_07_TX_{a..e}.min.gz
#         vision_rnd_05 ~/min-cost-flow/vision/vision_rnd_05_bone_subx_n6c100_a.min.gz
using MinimumCostFlows, SparseArrays, Serialization
using MinimumCostFlows: construct_constraint_matrix, Matrix2PerRowPM

name, files... = ARGS
prob = read_dimacs_mcf(length(files) == 1 ? files[1] : files, NativeMCFProblem{Int32, Int32})

A_lp, l = construct_constraint_matrix(prob, Float32)
A = SparseMatrixCSC{Float32}(A_lp)
println("A_lp: ", size(A), " nnz=", nnz(A))

outdir = joinpath(@__DIR__, "data")
mkpath(outdir)
serialize(joinpath(outdir, "$name.jls"), A)
serialize(joinpath(outdir, "$(name)_t.jls"), SparseMatrixCSC(transpose(A)))

# raw zoo formats: JDS arrays of A_lp (row map is Base.OneTo → identity) and
# the 2-per-row colidx of A_lpᵀ
@assert A_lp.row isa Base.OneTo
A_lpt = Matrix2PerRowPM(A_lp')
println("A_lpt: ", size(A_lpt))
serialize(joinpath(outdir, "$(name)_zoo.jls"),
          (jds_colidx=A_lp.colidx, jds_iterptr=A_lp.iterptr,
           jds_nrows=A_lp.nrows, jds_ncols=size(A_lp, 2),
           tpr_colidx=A_lpt.colidx, tpr_ncols=A_lpt.ncols))
println("EXPORT DONE")
