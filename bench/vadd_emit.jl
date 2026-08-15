# Step 2: build the vadd TTIR module via MLIR.jl (unregistered tt dialect,
# Reactant's generated tt builders) and print it for triton's IRSource.
using MLIR
using MLIR: IR, API

# Scaffold so Reactant's generated Triton.jl (which expects to live at
# Reactant.MLIR.Dialects.tt) resolves its relative imports against MLIR.jl.
module TTScaffold
using MLIR: MLIR
const IR = MLIR.IR
const API = MLIR.API
module Dialects
using ..IR: NamedAttribute
# triton (LLVM >= 17) spells these camelCase; MLIR.jl's own helpers use the
# old snake_case names, so shadow them here for the included tt builders.
operandsegmentsizes(segments) = NamedAttribute("operandSegmentSizes", Int32.(segments))
resultsegmentsizes(segments) = NamedAttribute("resultSegmentSizes", Int32.(segments))
include(joinpath(@__DIR__, "Triton.jl"))
end
end

const tt = TTScaffold.Dialects.tt
const arith = MLIR.Dialects.arith

ctx = IR.Context()
IR.activate(ctx)
IR.allow_unregistered_dialects!(true)
IR.load_all_available_dialects()

loc = IR.Location()
f32 = IR.Type(Float32)
i32 = IR.Type(Int32)
ptrf32 = IR.OpaqueType("tt", "ptr<f32>")
t_i32 = IR.TensorType([128], i32)
t_f32 = IR.TensorType([128], f32)
t_ptr = IR.TensorType([128], ptrf32)
t_i1 = IR.TensorType([128], IR.Type(Bool))

mod = IR.Module(loc)

reg = IR.Region()
entry = IR.Block([ptrf32, ptrf32, ptrf32, i32], [loc, loc, loc, loc])
push!(reg, entry)
xa, ya, oa, na = (IR.argument(entry, i) for i in 1:4)

IR.activate(entry)
try
    v(op) = IR.result(op, 1)
    c128 = v(arith.constant(; value=IR.Attribute(Int32(128)), result=i32))
    pid = v(tt.get_program_id(; result=i32, axis=IR.Attribute(Int32(0))))
    base = v(arith.muli(pid, c128; result=i32))
    range = v(tt.make_range(; result=t_i32, start=IR.Attribute(Int32(0)), end_=IR.Attribute(Int32(128))))
    offs = v(arith.addi(v(tt.splat(base; result=t_i32)), range; result=t_i32))
    nsplat = v(tt.splat(na; result=t_i32))
    mask = v(arith.cmpi(offs, nsplat; predicate=IR.Attribute(Int64(2)), result=t_i1))  # slt
    xp = v(tt.addptr(v(tt.splat(xa; result=t_ptr)), offs; result=t_ptr))
    yp = v(tt.addptr(v(tt.splat(ya; result=t_ptr)), offs; result=t_ptr))
    op = v(tt.addptr(v(tt.splat(oa; result=t_ptr)), offs; result=t_ptr))
    x = v(tt.load(xp, mask; result=t_f32))
    y = v(tt.load(yp, mask; result=t_f32))
    s = v(arith.addf(x, y; result=t_f32))
    tt.store(op, s, mask)
    tt.return_(IR.Value[])
finally
    IR.deactivate(entry)
end

ftype = IR.FunctionType([ptrf32, ptrf32, ptrf32, i32], [])
funcop = tt.func(;
    sym_name=IR.Attribute("vadd"),
    function_type=IR.Attribute(API.mlirTypeAttrGet(ftype)),
    sym_visibility=IR.Attribute("public"),
    body=reg,
)
push!(IR.body(mod), funcop)

text = string(mod)
print(text)
write(joinpath(@__DIR__, "vadd_julia.ttir"), text)
IR.deactivate(ctx)
println("=== wrote vadd_julia.ttir")
