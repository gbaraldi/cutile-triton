# TritonEmitter: lower cuTile StructuredIRCode to Triton IR (tt dialect) text.
#
#   Julia kernel + argtypes
#     → ct.code_structured(f, argtypes)     (cuTile front end, target-agnostic)
#     → emit_ttir(sci, argtypes)            (this file: MLIR.jl, unregistered tt)
#     → triton wheel IRSource compile       (compile_ttir.py)
#     → CUDA.jl launch                      (TritonRun.jl)
#
# Kernel ABI (must match TritonRun.abi_types / flatten_vals): each TileArray{T,N,I} becomes
# (ptr::!tt.ptr<T>, sizes::N×I, strides::N×I) in Julia dim order (I = i32 or
# i64, the TileArray's index type — cuTile types the gather offsets by it); plain
# scalars become one param; Constant{T,V} args are inlined. Triton appends
# global/profile scratch pointers at launch, not here.
#
# Shape convention: cuTile shapes are Julia column-major; Triton tensors are
# written row-major. Every shape/axis crossing the boundary is reversed once:
# Julia dim d (1-based) ↔ tensor axis N-d (0-based), so Julia dim 1 (stride 1)
# is the innermost tensor axis, which is what Triton's vectorizer wants.
module TritonEmitter

using MLIR
using MLIR: IR, API

module TTScaffold
using MLIR: MLIR
const IR = MLIR.IR
const API = MLIR.API
module Dialects
using ..IR: NamedAttribute
operandsegmentsizes(segments) = NamedAttribute("operandSegmentSizes", Int32.(segments))
resultsegmentsizes(segments) = NamedAttribute("resultSegmentSizes", Int32.(segments))
include(joinpath(@__DIR__, "dialects", "Triton.jl"))
end
end

const tt = TTScaffold.Dialects.tt
const arith = MLIR.Dialects.arith
const scf = MLIR.Dialects.scf
const math = MLIR.Dialects.math

using cuTile
const ct = cuTile
using IRStructurizer: BlockArgument, IfOp, ForOp, WhileOp, LoopOp, YieldOp,
                      ContinueOp, BreakOp, ConditionOp, Undef, StructuredIRCode
import IRStructurizer
const SBlock = IRStructurizer.Block
using Core: SSAValue, Argument, ReturnNode, PiNode
using Core.Compiler: widenconst

export emit_ttir, ArgSpec

# Upstream bugfix (JuliaGPU/cuTile.jl#336): promote_scalar_type does
# `T.parameters` after `T <: Tuple`, but a Union of tuples passes that subtype
# check and has no `.parameters` field → FieldError inside the front end (hit
# by hvcat kernels on cuTile 0.3.2). Guard Unions conservatively. This
# overwrite fixes it for both the Triton backend and native tileiras in this
# session.
# Upstream bugfix applied at runtime (evaluation into cuTile is not
# allowed during precompilation — including when a package extension
# precompiles with TileTriton loaded, so skip under jl_generating_output;
# every real session still runs this __init__ outside precompilation).
function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return
    @eval ct function promote_scalar_type(@nospecialize(T))
        T isa Union && return nothing
        T <: Number && return Tile{T, Tuple{}}
        if T <: Tuple
            params = T.parameters
            any(Base.isvarargtype, params) && return nothing
            any_promoted = false
            new_params = map(params) do P
                P = CC.widenconst(P)
                if P <: Number && !(P <: Tile)
                    any_promoted = true
                    Tile{P, Tuple{}}
                else
                    P
                end
            end
            any_promoted || return nothing
            return Tuple{new_params...}
        end
        return nothing
    end
    # Upstream bugfix (in-place rewrites removed by JuliaGPU/cuTile.jl#306,
    # after 1.0.0): the optimizer's one `inplace=true` rewrite rule — comparison strength reduction of
    # `cmpi(addi(a, addi(b, 1)), y, ≤, signed)` → `cmpi(addi(a, b), y, <)` in
    # NORMALIZATION_RULES — mutates the matched addi chain IN PLACE without
    # checking that it is single-use. When the 1-based index feeds both a mask
    # comparison and a gather/scatter (whose lowering subtracts 1), e.g.
    #     c = off .+ ct.arange(N); mask = c .≤ lim
    #     ct.gather(A, (c, cols); mask)
    # the shared `addi(off, addi(iota, 1))` is rewritten to 0-based for the
    # comparison, but the gather's `subi(c, 1)` still refers to it — every
    # gathered element shifts down by one row (hit by spmm_heavy_reduce_kernel;
    # whether it fires depends on worklist pop order, so e.g. an extra
    # `reshape` on the index tile masks it). Demoting the rule to the ordinary
    # non-inplace mode is always sound and keeps the optimization: the rewrite
    # then builds fresh ops and replaces only the matched root cmpi, leaving
    # shared sub-ops intact for their other users (DCE removes them when dead).
    for (i, r) in pairs(ct.NORMALIZATION_RULES)
        hasproperty(r, :inplace) && r.inplace || continue
        ct.NORMALIZATION_RULES[i] = ct.RewriteRule(r.lhs, r.rhs, r.guard, false)
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Value model
# ----------------------------------------------------------------------------

# A tensor/partition view, tracked at compile time (no IR of its own).
mutable struct ViewInfo
    ptr::IR.Value              # scalar !tt.ptr<T>
    sizes::Vector{Any}         # per Julia dim: IR.Value (i32) or Int
    strides::Vector{Any}
    elty::DataType
    tile_shape::Union{Nothing,Vector{Int}}   # set once it's a partition view
    padding::Symbol            # :undetermined | :zero | :negzero | :nan | :posinf | :neginf
    contiguous::Bool           # ArraySpec.Contiguous (Julia dim 1 has stride 1)
    desc::Union{Nothing,IR.Value}  # !tt.tensordesc — set when the TMA path is used
    order::Union{Nothing,Vector{Int}}  # dim permutation: tile dim d ↔ array dim order[d]
    step::Union{Nothing,Vector{Int}}   # tile-origin stride per tile dim (strided view); == tile_shape for partition views
end
ViewInfo(ptr, sizes, strides, elty, tile_shape, padding, contiguous, desc, order) =
    ViewInfo(ptr, sizes, strides, elty, tile_shape, padding, contiguous, desc, order, tile_shape)
view_step(view::ViewInfo, d::Int) = view.step === nothing ? view.tile_shape[d] : view.step[d]

"cuTile's padding mode as the view's padding symbol."
function padding_symbol(pm)
    s = Symbol(pm)
    s === :Zero ? :zero : s === :NegZero ? :negzero : s === :Nan ? :nan :
    s === :PosInf ? :posinf : s === :NegInf ? :neginf : :undetermined
end

"""
The value out-of-bounds elements read as (`nothing` when undetermined). As in
the native backend, only zero padding applies to non-float elements.
"""
function padding_value(pad::Symbol, elty)
    pad === :undetermined && return nothing
    pad === :zero && return elty <: AbstractFloat ? 0.0 : 0
    elty <: AbstractFloat ||
        error("padding_value $(Dict(:negzero => "neg_zero", :nan => "nan", :posinf => "pos_inf",
              :neginf => "neg_inf")[pad]) can only be used with floating point element types, got $elty")
    return pad === :negzero ? -0.0 : pad === :nan ? NaN : pad === :posinf ? Inf : -Inf
end

# A gather/scatter view: along `sparse_dim` the tile's coordinates are a 1-D
# tile of element indices, along every other dim a scalar element start plus
# the tile extent.
struct GSViewInfo
    base::ViewInfo        # the tensor view: ptr, sizes, strides
    shape::Vector{Int}
    sparse_dim::Int
    padding::Symbol
end

# Number of tile origins along dim d, cld(size, step), as cuTile's native
# index space counts them; a trailing window may be partial. The division
# runs at the size's width (an Int64-indexed dim can exceed 2^31 elements);
# the count itself is Int32, like cuTile's index space.
function index_space_extent(view::ViewInfo, d::Int)
    size_d = view.sizes[d]
    size_d isa IR.Value || return const_i32(cld(Int(size_d), view_step(view, d)))
    t = IR.type(size_d)
    return as_i32(v1(arith.ceildivsi(size_d, const_int(view_step(view, d), t); result=t)))
end

# Kernel-creation-time switch for the TMA/descriptor lowering (per-view
# legality still gates it; this is the opt-out).
const USE_TMA = Ref(true)

# Kernel-creation-time switch: in-tile offsets of every view at i64, for the
# launches whose spans (see `SpanTerm`) do not fit i32.
const WIDE_OFFSETS = Ref(false)


# TMA legality for a partition view: 2-D, unit-stride inner dim, box dims
# within CUtensorMap's 256-element limit, ≥16-byte inner box (hardware
# minimum — a 4-byte inner box traps the GPU). OOB zero-fill covers :zero
# and :undetermined directly; other paddings are recovered with an arithmetic
# select after the load (see emit_view_load). Dtype heuristic from measurement:
# descriptors are a large win for ≤16-bit element types and a regression for
# f32/tf32 on triton 3.7's Hopper pipeline, so gate on element size.
tma_eligible(N, tile_shape, pad, contiguous, elty) =
    USE_TMA[] && 2 <= N <= 5 && contiguous &&
    elty !== Bool &&    # Bool memory is i8-with-conversions, keep it off TMA
    sizeof(elty) <= 2 &&
    all(s -> 1 <= s <= 256, tile_shape) &&
    tile_shape[1] * sizeof(elty) >= 16

# Descriptor shapes are i32; sizes of an Int64-indexed array may not fit.
sizes_fit_i32(view::ViewInfo) = all(view.sizes) do s
    s isa IR.Value ? IR.type(s) == IR.Type(Int32) : typemin(Int32) <= s <= typemax(Int32)
end

"Emit tt.make_tensor_descriptor for a partition view (mirrors what Python's
tl.make_tensor_descriptor produces: i32 shape, i64 strides with literal-1
inner stride, row-major axis order)."
function emit_descriptor(view::ViewInfo)
    N = length(view.tile_shape)
    i32 = IR.Type(Int32)
    i64 = IR.Type(Int64)
    et = scalar_type(view.elty)
    shape_vals = IR.Value[]
    stride_vals = IR.Value[]
    for d in N:-1:1   # reversed (row-major) axis order
        s = view.sizes[d]
        push!(shape_vals, s isa IR.Value ? as_i32(s) : const_i32(Int(s)))
        if d == 1
            push!(stride_vals, const_scalar(1, i64))
        else
            st = view.strides[d]
            stv = st isa IR.Value ? st : const_i32(Int(st))
            push!(stride_vals, IR.type(stv) == i64 ? stv : v1(arith.extsi(stv; out=i64)))
        end
    end
    rev_sh = reverse(view.tile_shape)
    desc_t = IR.OpaqueType("tt", "tensordesc<" * join(rev_sh, "x") * "x" * string(et) * ">")
    return v1(tt.make_tensor_descriptor(view.ptr, shape_vals, stride_vals; result=desc_t))
end

# Value tagged as TF32 for tt.dot input precision.
struct TF32Val
    v::IR.Value
end

# env values: IR.Value | ViewInfo | TF32Val | Vector{Any} (tuple aggregate)
#           | Julia constants (Number, Bool, Tuple, DataType, enums, nothing)

struct ArgInfo
    ptr::IR.Value
    sizes::Vector{Any}
    strides::Vector{Any}
end

"Flattened kernel parameter description, for the launcher."
struct ArgSpec
    kind::Symbol       # :tilearray | :scalar
    elty::DataType
    ndims::Int         # 0 for scalars
    idxty::DataType    # TileArray index type (sizes/strides width); Int32 for scalars
end
ArgSpec(kind, elty, ndims) = ArgSpec(kind, elty, ndims, Int32)
"Index type parameter of a TileArray type (Int32 unless declared Int64)."
tilearray_indextype(@nospecialize(T)) =
    (T isa DataType && length(T.parameters) >= 3 && T.parameters[3] isa Type) ? T.parameters[3] : Int32

"""
One dimension of a pointer-path view whose in-tile offsets are i32: the tile
extent and origin step along it, and where its size and stride come from at
launch (an ABI parameter slot, or 0 for the constant beside it). The launcher
bounds the largest in-tile offset, sum((min(extent, size) - 1) * |stride|),
and the bounds check's last origin + extent with these.
"""
struct SpanTerm
    extent::Int
    step::Int
    size_slot::Int
    size::Int
    stride_slot::Int
    stride::Int
end

"KernelState handle: carries the implicit trailing seed parameter."
struct KSVal
    seed::IR.Value
end

mutable struct CG
    env::Dict{Int,Any}                 # SSAValue.id → value
    envtype::Dict{Int,Any}             # SSAValue.id → Julia type (for signedness)
    args::Dict{Int,Any}                # Argument.n → ArgInfo | scalar Value | constant
    blockargs::Dict{Int,Any}           # BlockArgument.id → Value (or nothing for tokens)
    seed::Union{Nothing,IR.Value}      # implicit KernelState seed param
    has_dot::Bool                      # kernel contains tt.dot (schedule heuristic)
    has_atomic::Bool                   # kernel contains atomics (autotune safety)
    params::Vector{IR.Value}           # entry block arguments, in ABI order
    spans::Vector{Vector{SpanTerm}}    # per i32-offset view, for the launch check
end
CG() = CG(Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), nothing, false, false,
          IR.Value[], Vector{SpanTerm}[])

"Julia eltype of an SSA operand, when tracked (signless MLIR needs this)."
function jl_eltype(cg::CG, @nospecialize(x), default::DataType)
    x isa SSAValue || return default
    t = get(cg.envtype, x.id, nothing)
    t === nothing && return default
    T = widenconst(t)
    return T <: ct.Tile ? tile_eltype(T) : T <: Number ? T : default
end

# ----------------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------------

function scalar_type(@nospecialize(T::DataType))
    if T <: Ptr
        return IR.OpaqueType("tt", "ptr<" * string(storage_type(eltype(T))) * ">")
    end
    T === Float64 && return IR.Type(Float64)
    T === Float32 && return IR.Type(Float32)
    T === Float16 && return IR.Type(Float16)
    T === Core.BFloat16 && return IR.Type(Core.BFloat16)
    T === ct.TFloat32 && return IR.Type(Float32)
    (T === Int64 || T === UInt64) && return IR.Type(Int64)
    (T === Int32 || T === UInt32) && return IR.Type(Int32)
    (T === Int16 || T === UInt16) && return IR.Type(Int16)
    (T === Int8 || T === UInt8) && return IR.Type(Int8)
    T === Bool && return IR.Type(Bool)
    error("scalar_type: unsupported $T")
end

# Bool tiles are i1 in registers but one byte in memory (matching Julia's
# Bool layout); triton cannot lower ptr<i1> loads/stores at all (LLVM
# castIsValid abort in translateModuleToLLVMIR, triton 3.7.1), so pointers
# always use the storage type and loads/stores convert (as Python triton does).
storage_type(@nospecialize(T::DataType)) = T === Bool ? IR.Type(Int8) : scalar_type(T)

# widen an i1 register tensor/scalar to its i8 storage form
function to_storage(v::IR.Value)
    t = IR.type(v)
    if API.mlirTypeIsARankedTensor(t)
        tensor_elem(t) == IR.Type(Bool) || return v
        return v1(arith.extui(v; out=IR.TensorType(tensor_shape(v), IR.Type(Int8))))
    end
    t == IR.Type(Bool) || return v
    return v1(arith.extui(v; out=IR.Type(Int8)))
end

# narrow an i8 storage tensor/scalar back to i1 (compare ≠ 0)
function to_register_i1(v::IR.Value)
    t = IR.type(v)
    if API.mlirTypeIsARankedTensor(t)
        z = materialize(0, t)
        return v1(arith.cmpi(v, z; predicate=IR.Attribute(Int64(1)),  # ne
                             result=IR.TensorType(tensor_shape(v), IR.Type(Bool))))
    end
    z = materialize(0, t)
    return v1(arith.cmpi(v, z; predicate=IR.Attribute(Int64(1)), result=IR.Type(Bool)))
end

tile_eltype(::Type{ct.Tile{T,S}}) where {T,S} = T
tile_shape(::Type{ct.Tile{T,S}}) where {T,S} = Int[S.parameters...]

# MLIR type for a cuTile Tile type: 0-D → scalar, N-D → tensor with reversed dims.
function tile_type(@nospecialize(T))
    T = widenconst(T)
    if T <: ct.Tile
        sh = tile_shape(T)
        et = scalar_type(tile_eltype(T))
        isempty(sh) && return et
        return IR.TensorType(reverse(sh), et)
    elseif T <: Number
        return scalar_type(T)
    end
    error("tile_type: unsupported $T")
end

tensor_of(sh_julia::Vector{Int}, et) = IR.TensorType(reverse(sh_julia), et)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

v1(op) = IR.result(op, 1)

# The MLIR C-API wrappers may return raw MlirType or an already-wrapped
# IR.Type depending on the MLIR.jl version in the manifest; normalize.
astype(x) = x isa IR.Type ? x : IR.Type(x)

"Element type of a shaped type; loud error (not a segfault) on scalars."
function tensor_elem(ty)
    t = astype(ty)
    API.mlirTypeIsAShaped(t) ||
        error("tensor_elem: expected a shaped type, got $(string(t))")
    return astype(API.mlirShapedTypeGetElementType(t))
end

function elem_attr(x, ty)
    ty == IR.Type(Bool) && return IR.Attribute(Bool(x), ty)
    IR.isinteger(ty) && return IR.Attribute(Int64(x), ty)
    API.mlirTypeIsAF64(ty) && return IR.Attribute(Float64(x))
    API.mlirTypeIsAF32(ty) && return IR.Attribute(Float32(x))
    API.mlirTypeIsAF16(ty) && return IR.Attribute(Float16(x))
    API.mlirTypeIsABF16(ty) && return IR.Attribute(Core.BFloat16(Float32(x)))
    error("elem_attr: unsupported type")
end

const_scalar(x, ty) = v1(arith.constant(; value=elem_attr(x, ty), result=ty))
const_i32(x::Integer) = const_scalar(x, IR.Type(Int32))

function splat_const(sh_julia::Vector{Int}, x, et)
    tty = tensor_of(sh_julia, et)
    attr = IR.Attribute(API.mlirDenseElementsAttrSplatGet(tty, elem_attr(x, et)))
    v1(arith.constant(; value=attr, result=tty))
end

is_tensor(v::IR.Value) = API.mlirTypeIsARankedTensor(IR.type(v))

"Coerce a scalar integer Value to i32 (indices may arrive as Julia Int64)."
function as_i32(v::IR.Value)
    IR.type(v) == IR.Type(Int32) && return v
    return v1(arith.trunci(v; out=IR.Type(Int32)))
end

"Coerce a scalar integer Value to the given integer type (trunc or sext)."
function as_int(v::IR.Value, ty)
    IR.type(v) == ty && return v
    ty == IR.Type(Int32) && return as_i32(v)
    return v1(arith.extsi(v; out=ty))
end

const_int(x::Integer, ty) = ty == IR.Type(Int32) ? const_i32(Int(x)) : const_scalar(Int64(x), ty)

"""
Offset type for a view's in-tile pointer arithmetic: i64 when any size/stride
is a 64-bit Value or a constant beyond Int32 (an Int64-indexed TileArray),
else i32. The tile origin goes into the base pointer at 64 bits (see
`view_ptrs_mask`), so only the span of one tile has to fit this type; the
launcher checks that it does (`SpanTerm`) and otherwise runs a build with
`WIDE_OFFSETS`.
"""
function view_offs_type(view::ViewInfo)
    i64 = IR.Type(Int64)
    WIDE_OFFSETS[] && return i64
    for xs in (view.sizes, view.strides), x in xs
        x isa IR.Value && IR.type(x) == i64 && return i64
        x isa Integer && abs(Int(x)) > typemax(Int32) && return i64
    end
    return IR.Type(Int32)
end

"""
Record the span of a view about to use i32 in-tile offsets, for the launch
check. Returns false when a size or stride is neither a kernel parameter nor
a constant, so the check could not evaluate it; the view then uses i64.
"""
function record_span!(cg::CG, view::ViewInfo)
    source(x) = x isa Integer ? (0, Int(x)) :
        let i = findfirst(==(x), cg.params); i === nothing ? nothing : (i, 0) end
    terms = SpanTerm[]
    for d in 1:length(view.tile_shape)
        ad = view.order === nothing ? d : view.order[d]
        sz = source(view.sizes[ad]); st = source(view.strides[ad])
        (sz === nothing || st === nothing) && return false
        push!(terms, SpanTerm(view.tile_shape[d], view_step(view, d), sz..., st...))
    end
    terms in cg.spans || push!(cg.spans, terms)
    return true
end

"Materialize an env value as an IR.Value of type `ty` (used for constants)."
function materialize(x, ty)::IR.Value
    x isa IR.Value && return x
    x isa TF32Val && return x.v
    if x isa Number || x isa Bool
        if API.mlirTypeIsARankedTensor(ty)
            et = tensor_elem(ty)
            n = API.mlirShapedTypeGetRank(ty)
            sh = Int[API.mlirShapedTypeGetDimSize(ty, i - 1) for i in 1:n]
            return splat_const(reverse(sh), x, et)
        end
        return const_scalar(x, ty)
    end
    if x isa Undef
        # Structurization artifact for a missing phi edge; the value is never
        # observed, so a zero of the right type is safe and always parses.
        return materialize(0, ty)
    end
    error("materialize: cannot materialize $x :: $(typeof(x))")
end

# ----------------------------------------------------------------------------
# Operand resolution
# ----------------------------------------------------------------------------

function resolve(cg::CG, @nospecialize(x))
    x isa SSAValue && return cg.env[x.id]
    x isa Argument && return cg.args[x.n]
    x isa BlockArgument && return cg.blockargs[x.id]
    x isa QuoteNode && return x.value
    x isa Undef && return x
    if x isa GlobalRef
        return isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : x.name
    end
    return x   # literal constant / enum instance / type / nothing
end

asvalue(x)::IR.Value = x isa TF32Val ? x.v : x::IR.Value

"Resolve to IR.Value, materializing constants with the type of a sibling value."
function resolve_typed(cg::CG, @nospecialize(x), sibling::IR.Value)
    r = resolve(cg, x)
    r isa IR.Value && return r
    r isa TF32Val && return r.v
    return materialize(r, IR.type(sibling))
end

# ----------------------------------------------------------------------------
# View load/store lowering: pointer + mask expansion
# ----------------------------------------------------------------------------

# Julia dim d of an N-D tile lives on tensor axis (N - d), 0-based.
axis_of(N, d) = N - d

"Broadcast a 1-D i32 tensor for Julia dim d to the full (reversed) tile shape."
function expand_to_full(offs1d::IR.Value, sh_julia::Vector{Int}, d::Int, et)
    N = length(sh_julia)
    N == 1 && return offs1d
    ax = axis_of(N, d)
    # insert size-1 dims at every axis except `ax`, then broadcast
    cur = offs1d
    cur_shape = Int[sh_julia[d]]
    # We need final reversed shape: rev = reverse(sh_julia)
    rev = reverse(sh_julia)
    # insert axes left-to-right
    for a in 0:(N - 1)
        a == ax && continue
        insert_at = a < ax ? a : a  # expand_dims inserts at position a
        newshape = copy(cur_shape)
        insert!(newshape, insert_at + 1, 1)
        cur = v1(tt.expand_dims(cur; result=IR.TensorType(newshape, et), axis=Int32(insert_at)))
        cur_shape = newshape
    end
    return v1(tt.broadcast(cur; result=IR.TensorType(rev, et)))
end

"""
Pointers and bounds mask (`nothing` without `check`) of a gather/scatter view
access. The sparse coordinates are data, which no launch-time span check can
bound, so the offsets are i64.
"""
function gs_ptrs_mask(view::GSViewInfo, idxs::Vector{Any}, check::Bool)
    sh = view.shape
    i64 = IR.Type(Int64)
    full_t = IR.TensorType(reverse(sh), i64)
    bt = IR.TensorType(reverse(sh), IR.Type(Bool))
    zero_f = materialize(0, full_t)
    offs = nothing
    mask = nothing
    for d in 1:length(sh)
        S = sh[d]
        idx = idxs[d]
        coord = if d == view.sparse_dim
            v = asvalue(idx)
            tensor_elem(IR.type(v)) == i64 ? v :
                v1(arith.extsi(v; out=IR.TensorType([S], i64)))
        else
            start = idx isa IR.Value ? as_int(idx, i64) : const_int(Int(idx), i64)
            rng = v1(tt.make_range(; result=IR.TensorType([S], IR.Type(Int32)),
                                   start=Int32(0), end_=Int32(S)))
            v1(arith.addi(v1(tt.splat(start; result=IR.TensorType([S], i64))),
                          v1(arith.extsi(rng; out=IR.TensorType([S], i64)));
                          result=IR.TensorType([S], i64)))
        end
        c = expand_to_full(coord, sh, d, i64)
        if check
            size_d = view.base.sizes[d]
            size_v = size_d isa IR.Value ? as_int(size_d, i64) : const_int(Int(size_d), i64)
            m = v1(arith.cmpi(c, v1(tt.splat(size_v; result=full_t));
                              predicate=IR.Attribute(Int64(2)), result=bt))   # slt
            if d == view.sparse_dim   # dense starts are asserted non-negative
                m = v1(arith.andi(m, v1(arith.cmpi(c, zero_f; predicate=IR.Attribute(Int64(5)),
                                                   result=bt)); result=bt))   # sge
            end
            mask = mask === nothing ? m : v1(arith.andi(mask, m; result=bt))
        end
        stride_d = view.base.strides[d]
        contrib = if stride_d isa Integer && stride_d == 1
            c
        else
            st = stride_d isa IR.Value ? as_int(stride_d, i64) : const_int(Int(stride_d), i64)
            v1(arith.muli(c, v1(tt.splat(st; result=full_t)); result=full_t))
        end
        offs = offs === nothing ? contrib : v1(arith.addi(offs, contrib; result=full_t))
    end
    ptr_t = IR.OpaqueType("tt", "ptr<" * string(storage_type(view.base.elty)) * ">")
    ptrs_t = IR.TensorType(reverse(sh), ptr_t)
    ptrs = v1(tt.addptr(v1(tt.splat(view.base.ptr; result=ptrs_t)), offs; result=ptrs_t))
    return ptrs, mask
end

"""
Tile indices for every dim of an N-D view: the missing trailing ones are 0, as
the native backend pads them. An index past its dim's tile count then selects
no element rather than wrapping into the next dim.
"""
function full_view_indices(view::ViewInfo, idxs::Vector{Any})
    N = length(view.tile_shape)
    length(idxs) >= N && return idxs
    return Any[idxs..., zeros(Int, N - length(idxs))...]
end

"""
Compute (ptrs, mask) for a partition-view access at tile index `idxs`
(0-based, per Julia dim). Returns (ptrs::Value, mask::Value or nothing).
"""
function view_ptrs_mask(view::ViewInfo, idxs::Vector{Any}, cg::CG)
    idxs = full_view_indices(view, idxs)
    sh = view.tile_shape
    N = length(sh)
    i32 = IR.Type(Int32)
    ot = view_offs_type(view)
    ot == i32 && !record_span!(cg, view) && (ot = IR.Type(Int64))
    et = storage_type(view.elty)
    ptr_t = IR.OpaqueType("tt", "ptr<" * string(et) * ">")

    # The tile origin, sum(start_d * stride_d), is folded into the base pointer
    # as one scalar i64: it is what overflows i32 for arrays beyond 2^31
    # elements, even when every size and stride fits 32 bits. The per-element
    # offsets within the tile stay at `ot` (i32 for Int32-indexed arrays), so
    # the access keeps the canonical splat(base) + i32-offsets form.
    i64 = IR.Type(Int64)
    base_off = nothing
    flat_offs = nothing
    mask = nothing
    for d in 1:N
        # order: tile dim d walks array dim ad (identity when unset)
        ad = view.order === nothing ? d : view.order[d]
        S = sh[d]
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_int(idx, ot) : const_int(Int(idx), ot)
        start = v1(arith.muli(idx_v, const_int(view_step(view, d), ot); result=ot))
        rng = v1(tt.make_range(; result=IR.TensorType([S], i32), start=Int32(0), end_=Int32(S)))
        ot == i32 || (rng = v1(arith.extsi(rng; out=IR.TensorType([S], ot))))
        rng_f = expand_to_full(rng, sh, d, ot)
        full_t = IR.type(rng_f)
        offs_f = v1(arith.addi(v1(tt.splat(start; result=full_t)), rng_f; result=full_t))

        # bounds mask along this dim: offs < size_ad
        size_d = view.sizes[ad]
        size_v = size_d isa IR.Value ? as_int(size_d, ot) : const_int(Int(size_d), ot)
        size_f = v1(tt.splat(size_v; result=full_t))
        m = v1(arith.cmpi(offs_f, size_f; predicate=IR.Attribute(Int64(2)), # slt
                          result=IR.TensorType(reverse(sh), IR.Type(Bool))))
        mask = mask === nothing ? m : v1(arith.andi(mask, m; result=IR.type(m)))

        # pointer: origin term into the scalar base, in-tile offsets * stride_ad
        # (skip the multiply for unit stride so AxisInfo sees contiguity)
        stride_d = view.strides[ad]
        unit = stride_d isa Int && stride_d == 1
        start64 = v1(arith.muli(as_int(idx_v, i64), const_int(view_step(view, d), i64); result=i64))
        term = unit ? start64 : v1(arith.muli(start64,
            stride_d isa IR.Value ? as_int(stride_d, i64) : const_int(Int(stride_d), i64); result=i64))
        base_off = base_off === nothing ? term : v1(arith.addi(base_off, term; result=i64))
        contrib = if unit
            rng_f
        else
            stride_v = stride_d isa IR.Value ? as_int(stride_d, ot) : const_int(Int(stride_d), ot)
            v1(arith.muli(rng_f, v1(tt.splat(stride_v; result=full_t)); result=full_t))
        end
        flat_offs = flat_offs === nothing ? contrib :
                    v1(arith.addi(flat_offs, contrib; result=full_t))
    end

    base = v1(tt.addptr(view.ptr, base_off; result=ptr_t))
    ptrs_t = IR.TensorType(reverse(sh), ptr_t)
    ptrs0 = v1(tt.splat(base; result=ptrs_t))
    ptrs = v1(tt.addptr(ptrs0, flat_offs; result=ptrs_t))
    return ptrs, mask
end

# Element offsets for descriptor access: idx_d * tile_d, reversed axis order.
function desc_offsets(view::ViewInfo, idxs::Vector{Any})
    idxs = full_view_indices(view, idxs)
    i32 = IR.Type(Int32)
    offs = IR.Value[]
    for d in length(view.tile_shape):-1:1
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_i32(idx) : const_i32(Int(idx))
        push!(offs, v1(arith.muli(idx_v, const_i32(view_step(view, d)); result=i32)))
    end
    return offs
end

"Just the in-bounds mask of a partition-view access (no pointer math)."
function view_bounds_mask(view::ViewInfo, idxs::Vector{Any})
    idxs = full_view_indices(view, idxs)
    sh = view.tile_shape
    N = length(sh)
    i32 = IR.Type(Int32)
    mask = nothing
    for d in 1:N
        ad = view.order === nothing ? d : view.order[d]
        S = sh[d]
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_i32(idx) : const_i32(Int(idx))
        start = v1(arith.muli(idx_v, const_i32(view_step(view, d)); result=i32))
        rng = v1(tt.make_range(; result=IR.TensorType([S], i32), start=Int32(0), end_=Int32(S)))
        offs1 = v1(arith.addi(v1(tt.splat(start; result=IR.TensorType([S], i32))), rng;
                              result=IR.TensorType([S], i32)))
        offs_f = expand_to_full(offs1, sh, d, i32)
        size_d = view.sizes[ad]
        size_v = size_d isa IR.Value ? as_i32(size_d) : const_i32(Int(size_d))
        size_f = v1(tt.splat(size_v; result=IR.type(offs_f)))
        m = v1(arith.cmpi(offs_f, size_f; predicate=IR.Attribute(Int64(2)),
                          result=IR.TensorType(reverse(sh), IR.Type(Bool))))
        mask = mask === nothing ? m : v1(arith.andi(mask, m; result=IR.type(m)))
    end
    return mask
end

function emit_view_load(view::ViewInfo, idxs::Vector{Any}, cg::CG)
    if view.desc !== nothing
        res_t = tensor_of(view.tile_shape, scalar_type(view.elty))
        loaded = v1(tt.descriptor_load(view.desc, desc_offsets(view, idxs); result=res_t))
        pad = padding_value(view.padding, view.elty)
        if pad !== nothing && !(pad == 0 && !signbit(pad))
            # TMA can only zero-fill OOB; recover any other padding with an
            # arithmetic select — data movement stays on TMA, the mask is
            # register-only index math (what Triton attention kernels do).
            m = view_bounds_mask(view, idxs)
            fill = splat_const(view.tile_shape, pad, scalar_type(view.elty))
            return v1(arith.select(m, loaded, fill; result=res_t))
        end
        return loaded
    end
    ptrs, mask = view_ptrs_mask(view, idxs, cg)
    res_t = tensor_of(view.tile_shape, storage_type(view.elty))
    pad = padding_value(view.padding, view.elty)
    loaded = pad === nothing ? v1(tt.load(ptrs, mask; result=res_t)) :
             v1(tt.load(ptrs, mask; other=splat_const(view.tile_shape, pad, storage_type(view.elty)),
                        result=res_t))
    return view.elty === Bool ? to_register_i1(loaded) : loaded
end

function emit_view_store(view::ViewInfo, val::IR.Value, idxs::Vector{Any}, cg::CG)
    if view.desc !== nothing
        tt.descriptor_store(view.desc, val, desc_offsets(view, idxs))
        return nothing
    end
    ptrs, mask = view_ptrs_mask(view, idxs, cg)
    tt.store(ptrs, view.elty === Bool ? to_storage(val) : val, mask)
    return nothing
end

# ----------------------------------------------------------------------------
# Predicates / enums
# ----------------------------------------------------------------------------

# triton RMWOp for an atomic op (:add, :max, ..., :xchg) on `jelt` elements
function rmw_code(op::Symbol, @nospecialize(jelt))
    op === :add ? (jelt <: AbstractFloat ? 5 : 4) :     # FADD : ADD
    op === :max ? (jelt <: Unsigned ? 8 : 6) :          # UMAX : MAX
    op === :min ? (jelt <: Unsigned ? 9 : 7) :          # UMIN : MIN
    op === :and ? 1 : op === :or ? 2 : op === :xor ? 3 :
    op === :xchg ? 10 : error("rmw_code: unknown atomic op :$op")
end

# triton MemSemantic: RELAXED=1 ACQUIRE=2 RELEASE=3 ACQUIRE_RELEASE=4
function mem_sem(x)
    s = Symbol(x)
    s === :Relaxed ? 1 : s === :Acquire ? 2 : s === :Release ? 3 : 4
end
# triton MemSyncScope: GPU=1 CTA=2 SYSTEM=3
function mem_scope(x)
    s = Symbol(x)
    (s === :Block || s === :CTA) ? 2 : s === :System ? 3 : 1
end

function cmpi_predicate(pred, signedness)
    p = Symbol(pred)
    signed = Symbol(signedness) === :Signed
    p === :Equal && return 0
    p === :NotEqual && return 1
    p === :LessThan && return signed ? 2 : 6
    p === :LessThanOrEqual && return signed ? 3 : 7
    p === :GreaterThan && return signed ? 4 : 8
    p === :GreaterThanOrEqual && return signed ? 5 : 9
    error("cmpi_predicate: unhandled $pred")
end

function cmpf_predicate(pred)
    p = Symbol(pred)
    p === :Equal && return 1              # oeq
    p === :GreaterThan && return 2        # ogt
    p === :GreaterThanOrEqual && return 3 # oge
    p === :LessThan && return 4           # olt
    p === :LessThanOrEqual && return 5    # ole
    p === :NotEqual && return 13          # une — Julia != is !(x == y), NaN-true
    error("cmpf_predicate: unhandled $pred")
end

# ----------------------------------------------------------------------------
# Statement walking
# ----------------------------------------------------------------------------

function walk_block!(cg::CG, blk::SBlock; at_top::Bool=false)
    _walk_stmts!(cg, collect(blk), 1, at_top)
end

function _walk_stmts!(cg::CG, stmts, i0::Int, at_top::Bool)
    i = i0
    while i <= length(stmts)
        idx, entry = stmts[i]
        st = entry.stmt
        if at_top && st isa IfOp && (st.then_region.terminator isa ReturnNode ||
                                     st.else_region.terminator isa ReturnNode)
            return emit_early_return_if!(cg, idx, st, stmts, i)
        end
        cg.env[idx] = walk_stmt!(cg, st, entry.type)
        cg.envtype[idx] = entry.type
        i += 1
    end
end

# Build a value-less scf region running `body` and yielding nothing.
function stmt_region(body::Function)
    region = IR.Region()
    b = IR.Block(IR.Type[], IR.Location[])
    push!(region, b)
    IR.activate(b)
    try
        body()
        scf.yield(IR.Value[])
    finally
        IR.deactivate(b)
    end
    return region
end

_not(c::IR.Value) = v1(arith.xori(c, materialize(true, IR.Type(Bool)); result=IR.Type(Bool)))

"A region whose body returns the values it yields."
function yield_region(body::Function)
    region = IR.Region()
    b = IR.Block(IR.Type[], IR.Location[])
    push!(region, b)
    IR.activate(b)
    try
        scf.yield(IR.Value[body()...])
    finally
        IR.deactivate(b)
    end
    return region
end

"""
Early `return` inside an `if` at kernel top level: scf has no early exit, so
predicate instead. The returning side's statements run under its condition;
the continuing side and *everything after the IfOp* run under the negation
(inside one scf region, so yielded values bind directly in the environment).
Only legal at top level — an early return inside a loop is rejected upstream
too.
"""
function emit_early_return_if!(cg::CG, idx::Int, op::IfOp, stmts, i::Int)
    c0 = resolve(cg, op.condition)
    cond = c0 isa IR.Value ? c0 : materialize(c0, IR.Type(Bool))
    tret = op.then_region.terminator isa ReturnNode
    eret = op.else_region.terminator isa ReturnNode
    if tret && eret
        tr = stmt_region(() -> walk_block!(cg, op.then_region; at_top=true))
        er = stmt_region(() -> walk_block!(cg, op.else_region; at_top=true))
        scf.if_(cond; results=IR.Type[], thenRegion=tr, elseRegion=er)
        return
    end
    retr = tret ? op.then_region : op.else_region
    contr = tret ? op.else_region : op.then_region
    c_ret = tret ? cond : _not(cond)
    scf.if_(c_ret; results=IR.Type[],
            thenRegion=stmt_region(() -> walk_block!(cg, retr; at_top=true)),
            elseRegion=stmt_region(() -> nothing))
    tr2 = stmt_region() do
        walk_block!(cg, contr; at_top=true)
        term = contr.terminator
        if term isa YieldOp && !isempty(term.values)
            cg.env[idx] = Any[resolve(cg, x) for x in term.values]
        end
        _walk_stmts!(cg, stmts, i + 1, true)
    end
    scf.if_(_not(c_ret); results=IR.Type[],
            thenRegion=tr2, elseRegion=stmt_region(() -> nothing))
    return
end

function walk_stmt!(cg::CG, @nospecialize(stmt), @nospecialize(typ))
    stmt isa Expr && return walk_expr!(cg, stmt, typ)
    stmt isa IfOp && return emit_if!(cg, stmt, typ)
    stmt isa ForOp && return emit_for!(cg, stmt, typ)
    stmt isa WhileOp && return emit_while!(cg, stmt, typ)
    stmt isa PiNode && return resolve(cg, stmt.val)
    stmt isa SSAValue && return cg.env[stmt.id]
    stmt isa BlockArgument && return cg.blockargs[stmt.id]
    stmt isa ct.MakeTokenNode && return nothing
    stmt isa ReturnNode && return nothing
    if nameof(typeof(stmt)) === :ThrowNode
        # a throw the program can reach at runtime fails an assertion, as in
        # the native backend; an unavoidable one is a compile-time error
        stmt.runtime || error(stmt.message)
        tt.assert(assert_once(materialize(false, IR.Type(Bool))); message=IR.Attribute(stmt.message))
        return nothing
    end
    stmt isa LoopOp && return emit_loop!(cg, stmt, typ)
    if stmt isa GlobalRef || stmt isa Number || stmt isa QuoteNode
        return resolve(cg, stmt)
    end
    # cuTile token nodes (JoinTokensNode etc.) — ignore
    pn = string(nameof(typeof(stmt)))
    occursin("Token", pn) && return nothing
    error("walk_stmt!: unhandled $(typeof(stmt)): $stmt")
end

function walk_expr!(cg::CG, e::Expr, @nospecialize(typ))
    if e.head === :call
        return walk_call!(cg, resolve(cg, e.args[1]), e.args[2:end], typ)
    elseif e.head === :invoke
        return walk_call!(cg, resolve(cg, e.args[2]), e.args[3:end], typ)
    elseif e.head === :boundscheck
        return false
    elseif e.head === :meta || e.head === :code_coverage_effect
        return nothing
    end
    error("walk_expr!: unhandled Expr head :$(e.head): $e")
end

# name of a callee for dispatch
function fname(@nospecialize(callee))
    callee isa Symbol && return callee
    return Symbol(nameof(callee))
end

const INT_BIN = Dict(:addi => arith.addi, :subi => arith.subi, :muli => arith.muli,
                     :andi => arith.andi, :ori => arith.ori, :xori => arith.xori,
                     :shli => arith.shli, :maxi => arith.maxsi, :mini => arith.minsi)
const FLT_BIN = Dict(:addf => arith.addf, :subf => arith.subf, :mulf => arith.mulf,
                     :divf => arith.divf, :maxf => arith.maximumf, :minf => arith.minimumf,
                     :remf => arith.remf)
const MATH_UN = Dict(:exp => math.exp, :exp2 => math.exp2, :log => math.log,
                     :log2 => math.log2, :sin => math.sin, :cos => math.cos,
                     :tan => math.tan, :sinh => math.sinh, :cosh => math.cosh,
                     :tanh => math.tanh, :sqrt => math.sqrt, :rsqrt => math.rsqrt,
                     :fabs => math.absf, :absf => math.absf, :absi => math.absi,
                     :floor => math.floor, :ceil => math.ceil, :erf => math.erf)
const MATH_BIN = Dict(:pow => math.powf, :atan2 => math.atan2)

function walk_call!(cg::CG, @nospecialize(callee), args::Vector{Any}, @nospecialize(typ))
    f = fname(callee)

    # ---- structural ----
    if f === :tuple
        return Any[resolve(cg, a) for a in args]
    elseif f === :getfield
        obj = resolve(cg, args[1])
        key = resolve(cg, args[2])
        if obj isa ArgInfo
            key === :ptr && return obj.ptr
            key === :sizes && return Any[obj.sizes...]
            key === :strides && return Any[obj.strides...]
            error("getfield(TileArray, :$key) unsupported")
        elseif obj isa KSVal
            key === :seed && return obj.seed
            error("getfield(KernelState, :$key) unsupported")
        elseif obj isa StructVal
            idx2 = key isa Symbol ? findfirst(==(key), fieldnames(obj.T)) : Int(key)
            idx2 === nothing && error("getfield: no field $key in $(obj.T)")
            return obj.fields[idx2]
        elseif obj isa Vector{Any}
            return obj[Int(key)]
        elseif obj isa Tuple
            return obj[Int(key)]
        elseif isbits(obj)   # a constant-folded immutable, e.g. a UnitRange
            return getfield(obj, key isa Symbol ? key : Int(key))
        end
        error("getfield: unhandled object $(typeof(obj))")

    # ---- grid ----
    elseif f === :get_tile_block_id
        axis = Int(resolve(cg, args[1]))
        return v1(tt.get_program_id(; result=IR.Type(Int32), axis=IR.Attribute(Int32(axis))))
    elseif f === :get_num_tile_blocks
        axis = Int(resolve(cg, args[1]))
        return v1(tt.get_num_programs(; result=IR.Type(Int32), axis=IR.Attribute(Int32(axis))))
    elseif f === :kernel_state
        cg.seed === nothing && error("kernel_state: seed param missing")
        return KSVal(cg.seed)

    # ---- views ----
    elseif f === :make_tensor_view
        TA = resolve(cg, args[1])
        ptr = resolve(cg, args[2])
        sizes = resolve(cg, args[3])
        strides = resolve(cg, args[4])
        elty = eltype(TA)
        contig = _spec_type(TA).parameters[3]::Bool
        return ViewInfo(asvalue(ptr), Any[sizes...], Any[strides...], elty,
                        nothing, :undetermined, contig, nothing, nothing)
    elseif f === :make_partition_view
        base = resolve(cg, args[1])::ViewInfo
        shape = resolve(cg, args[2])
        pad = padding_symbol(resolve(cg, args[3]))
        ord = length(args) >= 4 ? resolve(cg, args[4]) : nothing
        order = ord isa Tuple ? Int[ord...] : nothing
        view = ViewInfo(base.ptr, base.sizes, base.strides, base.elty,
                        Int[shape...], pad, base.contiguous, nothing, order)
        N = length(view.tile_shape)
        if order === nothing && tma_eligible(N, view.tile_shape, pad, view.contiguous, view.elty) &&
           sizes_fit_i32(view)
            view.desc = emit_descriptor(view)
        end
        return view
    elseif f === :make_strided_view
        # (tensor_view, tile_shape, step, padding, dim_map): like a partition
        # view, but tile origins advance by `step` instead of `tile_shape`
        base = resolve(cg, args[1])::ViewInfo
        shape = resolve(cg, args[2])
        step = resolve(cg, args[3])
        pad = padding_symbol(resolve(cg, args[4]))
        ord = length(args) >= 5 ? resolve(cg, args[5]) : nothing
        order = ord isa Tuple ? Int[ord...] : nothing
        view = ViewInfo(base.ptr, base.sizes, base.strides, base.elty,
                        Int[shape...], pad, base.contiguous, nothing, order, Int[step...])
        N = length(view.tile_shape)
        # descriptors only when the origins stay tile-aligned (plain partition)
        if order === nothing && view.step == view.tile_shape &&
           tma_eligible(N, view.tile_shape, pad, view.contiguous, view.elty) &&
           sizes_fit_i32(view)
            view.desc = emit_descriptor(view)
        end
        return view
    elseif f === :get_index_space_shape
        view = resolve(cg, args[1])::ViewInfo
        d = Int(resolve(cg, args[2])) + 1   # intrinsic axis is 0-based Julia dim
        return index_space_extent(view, d)
    elseif f === :load_partition_view || f === :load_strided_view
        view = resolve(cg, args[1])::ViewInfo
        idxs = resolve(cg, args[4])
        return emit_view_load(view, Any[idxs...], cg)
    elseif f === :make_gather_scatter_view
        # (tensor_view, shape, sparse_dim, padding_mode)
        pad = padding_symbol(resolve(cg, args[4]))
        return GSViewInfo(resolve(cg, args[1])::ViewInfo, Int[resolve(cg, args[2])...],
                          Int(resolve(cg, args[3])), pad)
    elseif f === :load_gather_scatter_view
        # (view, latency, allow_tma, indices, check_bounds, order, scope[, token])
        view = resolve(cg, args[1])::GSViewInfo
        ptrs, mask = gs_ptrs_mask(view, Any[resolve(cg, args[4])...], resolve(cg, args[5]) !== false)
        et = storage_type(view.base.elty)
        res_t = tensor_of(view.shape, et)
        pad = padding_value(view.padding, view.base.elty)
        loaded = mask === nothing || pad === nothing ? v1(tt.load(ptrs, mask; result=res_t)) :
                 v1(tt.load(ptrs, mask; other=splat_const(view.shape, pad, et), result=res_t))
        return view.base.elty === Bool ? to_register_i1(loaded) : loaded
    elseif f === :store_gather_scatter_view
        # (view, tile, latency, allow_tma, indices, check_bounds, order, scope[, token])
        view = resolve(cg, args[1])::GSViewInfo
        val = asvalue(resolve(cg, args[2]))
        ptrs, mask = gs_ptrs_mask(view, Any[resolve(cg, args[5])...], resolve(cg, args[6]) !== false)
        tt.store(ptrs, view.base.elty === Bool ? to_storage(val) : val, mask)
        return nothing
    elseif f === :store_partition_view || f === :store_strided_view
        view = resolve(cg, args[1])::ViewInfo
        val = asvalue(resolve(cg, args[2]))
        idxs = resolve(cg, args[5])
        return emit_view_store(view, val, Any[idxs...], cg)
    elseif f === :load_ptr_tko
        # gather: (ptrs, latency, mask, padding[, token])
        ptrs = asvalue(resolve(cg, args[1]))
        mask = length(args) >= 3 ? resolve(cg, args[3]) : nothing
        pad = length(args) >= 4 ? resolve(cg, args[4]) : nothing
        res_t = tile_type(typ)
        # Bool gathers load their i8 storage form and narrow at the end
        bool_mem = if API.mlirTypeIsARankedTensor(res_t)
            tensor_elem(res_t) == IR.Type(Bool)
        else
            res_t == IR.Type(Bool)
        end
        if bool_mem
            res_t = API.mlirTypeIsARankedTensor(res_t) ?
                IR.TensorType(Int[API.mlirShapedTypeGetDimSize(res_t, i - 1)
                                  for i in 1:API.mlirShapedTypeGetRank(res_t)], IR.Type(Int8)) :
                IR.Type(Int8)
        end
        mv = mask isa IR.Value ? mask : nothing
        # padding: an enum mode (Zero), a literal value, or a runtime scalar
        padval = pad isa IR.Value ? pad : pad isa Number ? pad :
                 (pad !== nothing && Symbol(pad) === :Zero) ? 0 : nothing
        if padval !== nothing && mv !== nothing
            other = if padval isa IR.Value
                is_tensor(padval) ? padval :
                API.mlirTypeIsARankedTensor(res_t) ? v1(tt.splat(padval; result=res_t)) : padval
            else
                materialize(padval, res_t)
            end
            loaded = v1(tt.load(ptrs, mv; other=other, result=res_t))
            return bool_mem ? to_register_i1(loaded) : loaded
        end
        loaded = mv === nothing ? v1(tt.load(ptrs; result=res_t)) :
                                  v1(tt.load(ptrs, mv; result=res_t))
        return bool_mem ? to_register_i1(loaded) : loaded
    elseif f === :store_ptr_tko
        # scatter: (ptrs, values, latency, mask[, token])
        ptrs = asvalue(resolve(cg, args[1]))
        val = to_storage(asvalue(resolve(cg, args[2])))   # no-op unless i1
        mask = length(args) >= 4 ? resolve(cg, args[4]) : nothing
        mv = mask isa IR.Value ? mask : nothing
        mv === nothing ? tt.store(ptrs, val) : tt.store(ptrs, val, mv)
        return nothing

    # ---- printing ----
    elseif f === :print_tko
        # (segments-and-values..., token): string segments become tt.print
        # prefixes; each value gets its own tt.print carrying the preceding text
        prefix = ""
        for a in args[1:end-1]   # drop the token
            it = resolve(cg, a)
            if it isa AbstractString
                prefix *= it
            elseif it isa IR.Value || it isa TF32Val
                vv = asvalue(it)
                t = IR.type(vv)
                et = API.mlirTypeIsARankedTensor(t) ? tensor_elem(t) : t
                # MLIR integers are signless: unsigned and Bool values must not
                # print sign-extended
                jt = jl_eltype(cg, a, Nothing)
                signed = API.mlirTypeIsAInteger(et) && (jt === Nothing || jt <: Signed)
                tt.print(IR.Value[vv]; prefix=prefix, hex=false, isSigned=Int32[signed])
                prefix = ""
            else
                prefix *= string(it)   # compile-time constant
            end
        end
        isempty(prefix) || tt.print(IR.Value[]; prefix=prefix, hex=false, isSigned=Int32[])
        return nothing

    # ---- constants / construction ----
    elseif f === :constant
        shape = resolve(cg, args[1])   # (shape, value, T)
        val = resolve(cg, args[2])
        T = resolve(cg, args[3])
        et = scalar_type(T)
        sh = Int[shape...]
        if val isa IR.Value || val isa TF32Val
            # runtime scalar: splat instead of a constant attribute
            vv = asvalue(val)
            isempty(sh) && return vv
            if is_tensor(vv)
                # a size-1 tile (e.g. fill(load(src, 1, (1,)), shape)):
                # tt.splat needs a scalar, so reshape to all-1 dims of the
                # target rank and tt.broadcast instead
                n = length(sh)
                vv = v1(tt.reshape(vv; result=IR.TensorType(fill(1, n), et)))
                return v1(tt.broadcast(vv; result=tensor_of(sh, et)))
            end
            return v1(tt.splat(vv; result=tensor_of(sh, et)))
        end
        isempty(sh) && return const_scalar(val, et)
        return splat_const(sh, val, et)
    elseif f === :iota
        # 1-D ascending 0..S-1; tt.make_range is i32-only, widen afterwards
        tty = tile_type(typ)
        S = Int(API.mlirShapedTypeGetDimSize(tty, 0))
        et = tensor_elem(tty)
        i32t = IR.TensorType([S], IR.Type(Int32))
        rng = v1(tt.make_range(; result=i32t, start=Int32(0), end_=Int32(S)))
        et == IR.Type(Int32) && return rng
        return v1(arith.extsi(rng; out=tty))
    elseif f === :broadcast
        src = resolve(cg, args[1])
        tty = tile_type(typ)
        if src isa IR.Value && is_tensor(src)
            # Julia broadcasting aligns leading (column-major) dims, i.e. the
            # trailing axes of the reversed tensor; prepend 1-dims up to rank.
            sv = src
            st = IR.type(sv)
            et = tensor_elem(st)
            m = Int(API.mlirShapedTypeGetRank(st))
            n = Int(API.mlirShapedTypeGetRank(tty))
            cur_sh = Int[API.mlirShapedTypeGetDimSize(st, i - 1) for i in 1:m]
            for _ in 1:(n - m)
                cur_sh = [1; cur_sh]
                sv = v1(tt.expand_dims(sv; result=IR.TensorType(cur_sh, et), axis=Int32(0)))
            end
            return v1(tt.broadcast(sv; result=tty))
        elseif !API.mlirTypeIsARankedTensor(tty)
            # 0-D broadcast: scalar in, scalar out
            return src isa IR.Value ? src : materialize(src, tty)
        else
            sv = src isa IR.Value ? src : materialize(src, tensor_elem(tty))
            return v1(tt.splat(sv; result=tty))
        end

    # ---- conversions ----
    elseif f === :ftof
        src = resolve(cg, args[1])
        Tdst = resolve(cg, args[2])
        if Tdst === ct.TFloat32
            return TF32Val(asvalue(src))
        end
        sv = asvalue(src)
        dst_et = scalar_type(Tdst)
        res_t = is_tensor(sv) ? tile_type(typ) : dst_et
        src_et = is_tensor(sv) ? tensor_elem(IR.type(sv)) : IR.type(sv)
        sw = API.mlirTypeIsAF64(src_et) ? 64 : API.mlirTypeIsAF32(src_et) ? 32 : 16
        dw = API.mlirTypeIsAF64(dst_et) ? 64 : API.mlirTypeIsAF32(dst_et) ? 32 : 16
        sw == dw && return sv
        return sw > dw ? v1(arith.truncf(sv; out=res_t)) : v1(arith.extf(sv; out=res_t))
    elseif f in (:itof, :ftoi, :exti, :trunci)
        r = resolve(cg, args[1])
        out_t = tile_type(typ)
        # constants fold directly into the target type
        (r isa IR.Value || r isa TF32Val) || return materialize(r, out_t)
        src = asvalue(r)
        IR.type(src) == out_t && return src  # same-width sign reinterpret: no-op
        signed = f === :trunci || Symbol(resolve(cg, args[3])) === :Signed
        op = f === :itof ? (signed ? arith.sitofp : arith.uitofp) :
             f === :ftoi ? (signed ? arith.fptosi : arith.fptoui) :
             f === :exti ? (signed ? arith.extsi : arith.extui) : arith.trunci
        return v1(op(src; out=out_t))
    elseif f === :to_scalar || f === :from_scalar
        return resolve(cg, args[1])

    # ---- comparisons ----
    elseif f === :cmpi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        pred = cmpi_predicate(resolve(cg, args[3]), resolve(cg, args[4]))
        return v1(arith.cmpi(av, bv; predicate=IR.Attribute(Int64(pred)), result=tile_type(typ)))
    elseif f === :cmpf
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        pred = cmpf_predicate(resolve(cg, args[3]))
        return v1(arith.cmpf(av, bv; predicate=IR.Attribute(Int64(pred)), result=tile_type(typ)))
    elseif f === :select
        c = asvalue(resolve(cg, args[1]))
        a = resolve(cg, args[2]); b = resolve(cg, args[3])
        av, bv = _pair_values(a, b)
        return v1(arith.select(c, av, bv; result=IR.type(av)))

    # ---- binary arithmetic ----
    elseif haskey(INT_BIN, f)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(INT_BIN[f](av, bv; result=IR.type(av)))
    elseif f === :maxf || f === :minf
        # optional 3rd arg propagate_nan (default false, i.e. NaN-ignoring)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        prop = length(args) >= 3 && resolve(cg, args[3]) == true
        op2 = f === :maxf ? (prop ? arith.maximumf : arith.maxnumf) :
                            (prop ? arith.minimumf : arith.minnumf)
        return v1(op2(av, bv; result=IR.type(av)))
    elseif haskey(FLT_BIN, f)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(FLT_BIN[f](av, bv; result=IR.type(av)))
    elseif f === :divi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(arith.divsi(av, bv; result=IR.type(av)))
    elseif f === :cldi || f === :fldi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return rounded_div(av, bv; ceil=f === :cldi, signed=Symbol(resolve(cg, args[3])) === :Signed)
    elseif f === :mulhii
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        # triton's own high-half multiply (arith.mul*i_extended isn't legalized)
        return v1(tt.mulhiui(av, bv; result=IR.type(av)))
    elseif f === :cat
        tiles = resolve(cg, args[1])
        vals = IR.Value[asvalue(t) for t in tiles]
        length(vals) >= 2 || return tiles[1]
        d = Int(resolve(cg, args[2])) + 1
        st = IR.type(vals[1])
        N = Int(API.mlirShapedTypeGetRank(st))
        et = tensor_elem(st)
        # pairwise join → move pair axis before the target axis → merge.
        # (order-preserving, unlike tt.cat, and layout-agnostic — tt.cat's
        # register-layout constraint rejects small i8 concats.)
        # i1 join/trans chains abort triton's LLVM translation (castIsValid
        # assert): do the shuffle in i8 and compare back to i1 at the end.
        was_i1 = et == IR.Type(Bool)
        if was_i1
            et = IR.Type(Int8)
            vals = IR.Value[v1(arith.extui(v; out=IR.TensorType(tensor_shape(v), et)))
                            for v in vals]
        end
        d <= 0 && (d = N + d)              # negative axis = from the end
        ax = axis_of(N, d)
        while length(vals) > 1
            length(vals) % 2 == 0 || error("cat: odd N-D concatenation unsupported")
            nxt = IR.Value[]
            for i in 1:2:length(vals) - 1
                x, y = vals[i], vals[i + 1]
                sh = tensor_shape(x)
                tensor_shape(y) == sh || error("cat: mismatched shapes")
                j = v1(tt.join(x, y; result=IR.TensorType([sh; 2], et)))
                order = [collect(0:ax-1); N; collect(ax:N-1)]
                t = _trans(j, order)
                msh = copy(sh); msh[ax + 1] *= 2
                push!(nxt, v1(tt.reshape(t; result=IR.TensorType(msh, et))))
            end
            vals = nxt
        end
        if was_i1
            r = vals[1]
            sh = tensor_shape(r)
            z = materialize(0, IR.type(r))
            return v1(arith.cmpi(r, z; predicate=IR.Attribute(Int64(1)),  # ne
                                 result=IR.TensorType(sh, IR.Type(Bool))))
        end
        return vals[1]
    elseif f === :remi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(arith.remsi(av, bv; result=IR.type(av)))
    elseif f === :negf
        # triton doesn't legalize arith.negf; match Python's `0 - x`
        a = resolve(cg, args[1])
        a isa Number && return -a
        a = asvalue(a)
        return v1(arith.subf(materialize(0, IR.type(a)), a; result=IR.type(a)))
    elseif f === :negi
        a = resolve(cg, args[1])
        a isa Number && return -a
        a = asvalue(a)
        return v1(arith.subi(materialize(0, IR.type(a)), a; result=IR.type(a)))
    elseif f === :shri
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        signed = Symbol(resolve(cg, args[3])) === :Signed
        op = signed ? arith.shrsi : arith.shrui
        return v1(op(av, bv; result=IR.type(av)))
    elseif f === :pow
        # triton's pipeline has no math.powf legalization; use exp2(y·log2 |x|)
        # plus IEEE-pow fixups: negative base with integral y takes the parity
        # sign (NaN when y is non-integral), and y == 0 is 1 for every x.
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        T = IR.type(av)
        bool_t = API.mlirTypeIsARankedTensor(T) ?
            IR.TensorType(tensor_shape(av), IR.Type(Bool)) : IR.Type(Bool)
        cmpfv(x, y, pred) = v1(arith.cmpf(x, y; predicate=IR.Attribute(Int64(pred)), result=bool_t))
        ax = v1(math.absf(av; result=T))
        lg = v1(math.log2(ax; result=T))
        pr = v1(arith.mulf(bv, lg; result=T))
        r = v1(math.exp2(pr; result=T))
        zero = materialize(0, T); one = materialize(1, T); two = materialize(2, T)
        isneg = cmpfv(av, zero, 4)                       # olt
        yfl = v1(math.floor(bv; result=T))
        isint = cmpfv(yfl, bv, 1)                        # oeq
        halfy = v1(arith.divf(bv, two; result=T))
        hfl = v1(math.floor(halfy; result=T))
        m = v1(arith.subf(bv, v1(arith.mulf(hfl, two; result=T)); result=T))
        isodd = cmpfv(m, one, 1)                         # y - 2⌊y/2⌋ == 1
        negr = v1(arith.subf(zero, r; result=T))
        signed_r = v1(arith.select(isodd, negr, r; result=T))
        nanv = materialize(NaN, T)
        neg_case = v1(arith.select(isint, signed_r, nanv; result=T))
        res = v1(arith.select(isneg, neg_case, r; result=T))
        iszero_y = cmpfv(bv, zero, 1)
        return v1(arith.select(iszero_y, one, res; result=T))
    elseif haskey(MATH_UN, f)
        a = asvalue(resolve(cg, args[1]))
        return v1(MATH_UN[f](a; result=IR.type(a)))
    elseif haskey(MATH_BIN, f)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(MATH_BIN[f](av, bv; result=IR.type(av)))
    elseif f === :fma
        a = asvalue(resolve(cg, args[1])); b = asvalue(resolve(cg, args[2])); c = asvalue(resolve(cg, args[3]))
        return v1(math.fma(a, b, c; result=IR.type(a)))

    # ---- matmul ----
    elseif f === :mma
        cg.has_dot = true
        a = resolve(cg, args[1]); b = resolve(cg, args[2]); acc = asvalue(resolve(cg, args[3]))
        tf32 = a isa TF32Val || b isa TF32Val
        av = asvalue(a); bv = asvalue(b)
        aet = tensor_elem(IR.type(av))
        if IR.isinteger(aet)
            # Triton's int8 dot needs K ≥ 32 and is signed-only (the Python
            # frontend asserts; via IRSource it silently miscompiles). Emulate
            # with an exact IEEE f32 dot: exact while K·max|a·b| < 2²⁴, i.e.
            # K ≤ 256 for 8-bit operands.
            K = tensor_shape(av)[end]
            w = Int(API.mlirIntegerTypeGetWidth(aet))
            (w <= 8 && K <= 256) ||
                error("mma: integer dot emulation supports 8-bit operands with K ≤ 256 (got i$w, K=$K)")
            f32 = IR.Type(Float32)
            cvt(v, x) = v1((jl_eltype(cg, x, Int8) <: Unsigned ? arith.uitofp : arith.sitofp)(
                v; out=IR.TensorType(tensor_shape(v), f32)))
            afv = cvt(av, args[1]); bfv = cvt(bv, args[2])
            accf = v1(arith.sitofp(acc; out=IR.TensorType(tensor_shape(acc), f32)))
            df = v1(tt.dot(afv, bfv, accf; d=IR.type(accf), inputPrecision=IR.Attribute(Int32(2))))
            return v1(arith.fptosi(df; out=IR.type(acc)))
        end
        attrs = tf32 ? (; inputPrecision=IR.Attribute(Int32(0))) : (;)
        return v1(tt.dot(av, bv, acc; d=IR.type(acc), attrs...))

    # ---- atomics ----
    elseif f in (:atomic_add, :atomic_max, :atomic_min, :atomic_and, :atomic_or,
                 :atomic_xor, :atomic_xchg)
        # (ptr_tile, val, mask, memory_order, memory_scope[, token])
        cg.has_atomic = true
        ptrs = asvalue(resolve(cg, args[1]))
        val = materialize(resolve(cg, args[2]), tile_type(typ))
        mask = resolve(cg, args[3])
        # a constant-false mask touches no memory (the result is unspecified)
        mask === false && return materialize(0, tile_type(typ))
        mv = mask isa IR.Value ? mask : nothing
        rmw = rmw_code(Symbol(string(f)[length("atomic_")+1:end]), tile_eltype(widenconst(typ)))
        return v1(tt.atomic_rmw(ptrs, val, mv; result=tile_type(typ),
                                atomic_rmw_op=IR.Attribute(Int32(rmw)),
                                sem=IR.Attribute(Int32(mem_sem(resolve(cg, args[4])))),
                                scope=IR.Attribute(Int32(mem_scope(resolve(cg, args[5]))))))
    elseif f in (:atomic_red_view_add, :atomic_red_view_max, :atomic_red_view_min,
                 :atomic_red_view_or, :atomic_red_view_and, :atomic_red_view_xor)
        # (view, tile, indices[, token]): relaxed, device-scope reduction of a
        # tile into the view's tile at `indices`; returns nothing
        cg.has_atomic = true
        view = resolve(cg, args[1])::ViewInfo
        val = asvalue(resolve(cg, args[2]))
        idxs = resolve(cg, args[3])
        rmw = rmw_code(Symbol(string(f)[length("atomic_red_view_")+1:end]),
                       jl_eltype(cg, args[2], view.elty))
        ptrs, mask = view_ptrs_mask(view, Any[(idxs isa Vector ? idxs : (idxs,))...], cg)
        tt.atomic_rmw(ptrs, val, mask; result=IR.type(val), atomic_rmw_op=IR.Attribute(Int32(rmw)),
                      sem=IR.Attribute(Int32(1)), scope=IR.Attribute(Int32(1)))   # RELAXED, GPU
        return nothing
    elseif f === :atomic_cas
        # (ptr_tile, expected, desired, mask, memory_order, memory_scope[, token])
        cg.has_atomic = true
        mask = resolve(cg, args[4])
        ptrs = asvalue(resolve(cg, args[1]))
        rt = tile_type(typ)
        cmp = materialize(resolve(cg, args[2]), rt)
        val = materialize(resolve(cg, args[3]), rt)
        sem = IR.Attribute(Int32(mem_sem(resolve(cg, args[5]))))
        scope = IR.Attribute(Int32(mem_scope(resolve(cg, args[6]))))
        cas(p, c, v) = v1(tt.atomic_cas(p, c, v; result=rt, sem, scope))
        # masked-out elements are untouched and return `expected` (cuTile's contract)
        mask === false && return cmp
        mask isa IR.Value || return cas(ptrs, cmp, val)
        return masked_cas(cas, ptrs, cmp, val, mask, rt)

    # ---- reductions ----
    elseif f === :reduce
        tiles = resolve(cg, args[1])
        srcs = IR.Value[asvalue(t) for t in tiles]
        d = Int(resolve(cg, args[2])) + 1      # intrinsic axis is 0-based Julia dim
        combiner = resolve(cg, args[3])
        src_t = IR.type(srcs[1])
        N = Int(API.mlirShapedTypeGetRank(src_t))
        ax = axis_of(N, d)
        ets = [tensor_elem(IR.type(s)) for s in srcs]
        rev_sh = Int[API.mlirShapedTypeGetDimSize(src_t, i - 1) for i in 1:N]
        red_sh = [rev_sh[i] for i in 1:N if i - 1 != ax]
        red_ts = IR.Type[isempty(red_sh) ? et : IR.TensorType(red_sh, et) for et in ets]

        RT = widenconst(typ)
        jelts = DataType[tile_eltype(widenconst(p)) for p in (RT <: Tuple ? RT.parameters : (RT,))]
        region = combiner_region(combiner, jelts, ets, tt.reduce_return)
        red = tt.reduce(srcs; result=red_ts, axis=IR.Attribute(Int32(ax)), combineOp=region)
        # cuTile keeps the reduced dim (size 1); restore it. A 1-D reduce
        # yields a scalar, which expand_dims can't take — splat instead.
        keep_sh = copy(rev_sh); keep_sh[ax + 1] = 1
        outs = Any[]
        for (k, et) in enumerate(ets)
            rv = IR.result(red, k)
            keep_t = IR.TensorType(keep_sh, et)
            push!(outs, isempty(red_sh) ? v1(tt.splat(rv; result=keep_t)) :
                        v1(tt.expand_dims(rv; result=keep_t, axis=Int32(ax))))
        end
        return outs
    elseif f === :scan
        # (tiles_tuple, axis0, combiner, identities, reverse)
        tiles = resolve(cg, args[1])
        length(tiles) == 1 || error("scan: only single-tile scans supported")
        src = asvalue(tiles[1])
        d = Int(resolve(cg, args[2])) + 1
        combiner = resolve(cg, args[3])
        rev = resolve(cg, args[5]) === true
        src_t = IR.type(src)
        N = Int(API.mlirShapedTypeGetRank(src_t))
        ax = axis_of(N, d)
        et = tensor_elem(src_t)
        RT = widenconst(typ)
        jelt = RT <: Tuple ? tile_eltype(widenconst(RT.parameters[1])) : tile_eltype(RT)
        region = combiner_region(combiner, [jelt], [et], tt.scan_return)
        op = tt.scan([src]; result=[src_t], axis=IR.Attribute(Int32(ax)),
                     reverse=IR.Attribute(rev), combineOp=region)
        return Any[IR.result(op, 1)]

    # ---- pointer arithmetic (raw gather/scatter-style access) ----
    elseif f === :offset
        base = resolve(cg, args[1])
        off = resolve(cg, args[2])
        res_t = tile_type(typ)
        bv = base isa IR.Value ? base : error("offset: unresolved base")
        ov = off isa IR.Value ? off :
             const_scalar(Int(off), IR.Type(Int32))
        # broadcast scalar/tile mismatches to the result shape
        if API.mlirTypeIsARankedTensor(res_t)
            if !is_tensor(bv)
                bv = v1(tt.splat(bv; result=res_t))
            end
            if !is_tensor(ov)
                et = tensor_elem(IR.type(bv))
                n = API.mlirShapedTypeGetRank(res_t)
                sh = Int[API.mlirShapedTypeGetDimSize(res_t, i - 1) for i in 1:n]
                ov = v1(tt.splat(ov; result=IR.TensorType(sh, IR.type(ov))))
            end
        end
        return v1(tt.addptr(bv, ov; result=res_t))
    elseif f === :getglobal
        return getglobal(resolve(cg, args[1]), resolve(cg, args[2]))
    elseif f === :reshape
        src = resolve(cg, args[1])
        tty = tile_type(typ)
        if !(src isa IR.Value)
            return materialize(src, tty)
        elseif !is_tensor(src)
            return API.mlirTypeIsARankedTensor(tty) ? v1(tt.splat(src; result=tty)) : src
        elseif !API.mlirTypeIsARankedTensor(tty)
            # tensor (single element) → scalar: sum-reduce all axes
            sv = src
            while is_tensor(sv)
                st = IR.type(sv)
                et = tensor_elem(st)
                n = Int(API.mlirShapedTypeGetRank(st))
                sh = Int[API.mlirShapedTypeGetDimSize(st, i - 1) for i in 1:n]
                red_sh = sh[2:end]
                red_t = isempty(red_sh) ? et : IR.TensorType(red_sh, et)
                region = IR.Region()
                blk = IR.Block([et, et], [IR.Location(), IR.Location()])
                push!(region, blk)
                IR.activate(blk)
                try
                    x = IR.argument(blk, 1); y = IR.argument(blk, 2)
                    isflt = !IR.isinteger(et)
                    c = isflt ? v1(arith.addf(x, y; result=et)) : v1(arith.addi(x, y; result=et))
                    tt.reduce_return([c])
                finally
                    IR.deactivate(blk)
                end
                sv = IR.result(tt.reduce([sv]; result=[red_t], axis=IR.Attribute(Int32(0)), combineOp=region), 1)
            end
            return sv
        else
            return v1(tt.reshape(src; result=tty))
        end
    elseif f === :bitcast
        src = resolve(cg, args[1])
        tty = tile_type(typ)
        (src isa IR.Value || src isa TF32Val) || return materialize(src, tty)
        sv = asvalue(src)
        IR.type(sv) == tty && return sv
        return is_tensor(sv) ? v1(tt.bitcast(sv; result=tty)) :
                               v1(arith.bitcast(sv; out=tty))
    elseif f === :extract
        src = asvalue(resolve(cg, args[1]))
        idx = resolve(cg, args[2])   # 0-based, tile-granular, per Julia dim
        shp = resolve(cg, args[3])
        any(i -> i isa IR.Value, idx) &&
            error("extract: runtime slice indices not supported yet")
        N = length(shp)
        v = src
        for d in 1:N
            S = Int(shp[d])
            ax = axis_of(N, d)
            sh = tensor_shape(v)
            sh[ax + 1] == S && Int(idx[d]) == 0 && continue
            # move axis last, slice, move back
            n = length(sh)
            if ax != n - 1
                order = collect(0:n-1); order[ax + 1], order[n] = order[n], order[ax + 1]
                v = _trans(v, order)
                v = extract_last_axis(v, Int(idx[d]) * S, S)
                v = _trans(v, order)
            else
                v = extract_last_axis(v, Int(idx[d]) * S, S)
            end
        end
        return v
    elseif f === :pack
        # Tile{S,(N)} → Tile{UInt8,(N·bw/8)}: repeatedly split each element
        # into (lo, hi) half-width parts, interleaved little-endian.
        v = asvalue(resolve(cg, args[1]))
        st = IR.type(v)
        et = tensor_elem(st)
        bw = Int(API.mlirIntegerTypeGetWidth(et))
        n = Int(API.mlirShapedTypeGetDimSize(st, 0))
        while bw > 8
            hw = bw ÷ 2
            ht = IR.Type(hw == 32 ? Int32 : hw == 16 ? Int16 : Int8)
            cur_t = IR.TensorType([n], tensor_elem(IR.type(v)))
            lo = v1(arith.trunci(v; out=IR.TensorType([n], ht)))
            sh_amt = materialize(hw, IR.type(v))
            hi = v1(arith.trunci(v1(arith.shrui(v, sh_amt; result=IR.type(v))); out=IR.TensorType([n], ht)))
            j = v1(tt.join(lo, hi; result=IR.TensorType([n, 2], ht)))
            v = v1(tt.reshape(j; result=IR.TensorType([2n], ht)))
            n *= 2; bw = hw
        end
        return v
    elseif f === :unpack
        # Tile{UInt8,(N)} → Tile{T,(N·8/bw)}: inverse of pack.
        v = asvalue(resolve(cg, args[1]))
        T = resolve(cg, args[2])
        tgt_bw = 8 * sizeof(T)
        n = Int(API.mlirShapedTypeGetDimSize(IR.type(v), 0))
        bw = 8
        while bw < tgt_bw
            dw = 2bw
            dt = IR.Type(dw == 64 ? Int64 : dw == 32 ? Int32 : Int16)
            cur_et = tensor_elem(IR.type(v))
            r = v1(tt.reshape(v; result=IR.TensorType([n ÷ 2, 2], cur_et)))
            s = tt.split(r; outLHS=IR.TensorType([n ÷ 2], cur_et), outRHS=IR.TensorType([n ÷ 2], cur_et))
            wt = IR.TensorType([n ÷ 2], dt)
            lo = v1(arith.extui(IR.result(s, 1); out=wt))
            hi = v1(arith.extui(IR.result(s, 2); out=wt))
            hi = v1(arith.shli(hi, materialize(bw, wt); result=wt))
            v = v1(arith.ori(lo, hi; result=wt))
            n ÷= 2; bw = dw
        end
        return v
    elseif f === :permute
        src = asvalue(resolve(cg, args[1]))
        perm = resolve(cg, args[2])   # 0-based Julia dims
        N = length(perm)
        order = Int32[N - 1 - Int(perm[N - a]) for a in 0:(N - 1)]
        return v1(tt.trans(src; result=tile_type(typ), order=order))

    # ---- misc / dropped ----
    elseif f === :assume
        return resolve(cg, args[1])
    elseif f === :assert
        cond = resolve(cg, args[1])
        cond === true && return nothing   # front end already proved it
        msg = resolve(cg, args[2])
        cv = cond isa IR.Value ? cond : materialize(cond, IR.Type(Bool))
        tt.assert(assert_once(cv); message=IR.Attribute(String(msg)))
        return nothing
    elseif f === :MakeTokenNode || f === :JoinTokensNode
        return nothing
    elseif f === :fpmode_begin || f === :fpmode_end
        # fast-math mode scoping: Triton has no region-scoped fp modes; ops
        # default to its standard fast-math behavior. Accept and ignore.
        return nothing
    elseif f === :Tile && length(args) == 1
        # 0-D tile of a scalar: Triton has no rank-0 tensors; scalars already
        # broadcast against tiles wherever a 0-D tile would.
        return resolve(cg, args[1])
    end

    error("walk_call!: unhandled intrinsic $f (args=$args)")
end

# Pair two operands, materializing constants. When both are constants (heavy
# const-folding, e.g. Philox seed math), type them from their Julia types.
function _pair_values(a, b)
    if a isa IR.Value || a isa TF32Val
        av = asvalue(a)
        bv = (b isa IR.Value || b isa TF32Val) ? asvalue(b) : materialize(b, IR.type(av))
        return av, bv
    elseif b isa IR.Value || b isa TF32Val
        bv = asvalue(b)
        av = materialize(a, IR.type(bv))
        return av, bv
    else
        av = materialize(a, scalar_type(typeof(a)))
        bv = materialize(b, IR.type(av))
        return av, bv
    end
end

# A scalar assert fails in every thread at once, and with several warps in
# several programs failing together the driver prints none of the messages.
# Fold the condition into a 1024-element tensor whose element 0 alone carries
# it: 1024 is the most threads a program has, so whatever the layout, one
# thread per program reports, as the native backend does.
function assert_once(cv::IR.Value)
    is_tensor(cv) && return cv
    n = 1024
    i32 = IR.Type(Int32)
    bt = IR.TensorType([n], IR.Type(Bool))
    r = v1(tt.make_range(; result=IR.TensorType([n], i32), start=Int32(0), end_=Int32(n)))
    others = v1(arith.cmpi(r, splat_const([n], 0, i32); predicate=IR.Attribute(Int64(1)),  # ne
                           result=bt))
    return v1(arith.ori(v1(tt.splat(cv; result=bt)), others; result=bt))
end

# tt.atomic_cas has no mask. Masked-out lanes compare-and-swap 0 for 0 at an
# active lane's address, which leaves memory unchanged whatever order the
# atomics take, and return `expected`; with no active lane the CAS is skipped.
function masked_cas(cas, ptrs::IR.Value, cmp::IR.Value, val::IR.Value, mask::IR.Value, rt)
    if !is_tensor(ptrs)
        return v1(scf.if_(mask; results=IR.Type[rt],
                          thenRegion=yield_region(() -> (cas(ptrs, cmp, val),)),
                          elseRegion=yield_region(() -> (cmp,))))
    end
    addr = max_active_addr(ptrs, mask)
    i64 = IR.Type(Int64)
    any = v1(arith.cmpi(addr, materialize(0, i64); predicate=IR.Attribute(Int64(1)),  # ne
                        result=IR.Type(Bool)))
    pt = IR.type(ptrs)
    zero = materialize(0, rt)
    function active()
        p0 = v1(tt.int_to_ptr(addr; result=tensor_elem(pt)))
        safe = v1(arith.select(mask, ptrs, v1(tt.splat(p0; result=pt)); result=pt))
        old = cas(safe, v1(arith.select(mask, cmp, zero; result=rt)),
                  v1(arith.select(mask, val, zero; result=rt)))
        return (v1(arith.select(mask, old, cmp; result=rt)),)
    end
    return v1(scf.if_(any; results=IR.Type[rt], thenRegion=yield_region(active),
                      elseRegion=yield_region(() -> (cmp,))))
end

"The largest address among the lanes `mask` keeps, as an i64 (0 if none)."
function max_active_addr(ptrs::IR.Value, mask::IR.Value)
    sh = tensor_shape(ptrs)
    i64 = IR.Type(Int64)
    it = IR.TensorType(sh, i64)
    a = v1(arith.select(mask, v1(tt.ptr_to_int(ptrs; result=it)), materialize(0, it); result=it))
    length(sh) == 1 || (a = v1(tt.reshape(a; result=IR.TensorType([prod(sh)], i64))))
    region = combiner_region(max, DataType[UInt64], IR.Type[i64], tt.reduce_return)
    return IR.result(tt.reduce([a]; result=[i64], axis=IR.Attribute(Int32(0)), combineOp=region), 1)
end

# Floor/ceil division. The AMD backend leaves arith.floordivsi/ceildivsi/
# ceildivui on tensors unlowered, so tensors get a truncating divide plus a
# ±1 correction where the remainder is nonzero.
function rounded_div(av::IR.Value, bv::IR.Value; ceil::Bool, signed::Bool)
    t = IR.type(av)
    if !is_tensor(av)
        op = ceil ? (signed ? arith.ceildivsi : arith.ceildivui) :
                    (signed ? arith.floordivsi : arith.divui)
        return v1(op(av, bv; result=t))
    end
    q = v1((signed ? arith.divsi : arith.divui)(av, bv; result=t))
    !signed && !ceil && return q
    bt = IR.TensorType(tensor_shape(av), IR.Type(Bool))
    cmp(x, y, pred) = v1(arith.cmpi(x, y; predicate=IR.Attribute(Int64(pred)), result=bt))
    r = v1((signed ? arith.remsi : arith.remui)(av, bv; result=t))
    zero = materialize(0, t)
    adj = cmp(r, zero, 1)                                     # ne
    if signed
        # truncation rounded toward zero: off by one when the exact quotient
        # is negative (floor) or positive (ceil); r has the sign of a
        off = cmp(cmp(r, zero, 2), cmp(bv, zero, 2), ceil ? 0 : 1)   # slt, then eq / ne
        adj = v1(arith.andi(adj, off; result=bt))
    end
    q1 = v1((ceil ? arith.addi : arith.subi)(q, materialize(1, t); result=t))
    return v1(arith.select(adj, q1, q; result=t))
end

# ----------------------------------------------------------------------------
# Tile slicing: Triton has no slice op; a power-of-2-aligned slice is a chain
# of contiguous halvings, each halving = reshape (…,L)→(…,2,L/2), transpose the
# trailing pair, tt.split, pick a half.
# ----------------------------------------------------------------------------

tensor_shape(v::IR.Value) = begin
    t = IR.type(v)
    Int[API.mlirShapedTypeGetDimSize(t, i - 1) for i in 1:API.mlirShapedTypeGetRank(t)]
end

function _trans(v::IR.Value, order::Vector{Int})
    sh = tensor_shape(v)
    et = tensor_elem(IR.type(v))
    nsh = [sh[o + 1] for o in order]
    v1(tt.trans(v; result=IR.TensorType(nsh, et), order=Int32.(order)))
end

function halve(v::IR.Value, high::Bool)
    sh = tensor_shape(v)
    et = tensor_elem(IR.type(v))
    L = sh[end]
    r1 = v1(tt.reshape(v; result=IR.TensorType([sh[1:end-1]; 2; L ÷ 2], et)))
    n = length(sh) + 1
    order = [collect(0:n-3); n - 1; n - 2]
    r2 = _trans(r1, order)
    half_t = IR.TensorType([sh[1:end-1]; L ÷ 2], et)
    s = tt.split(r2; outLHS=half_t, outRHS=half_t)
    return IR.result(s, high ? 2 : 1)
end

"Extract `S` elements starting at element offset `o` along the LAST axis."
function extract_last_axis(v::IR.Value, o::Int, S::Int)
    L = tensor_shape(v)[end]
    while L > S
        h = L ÷ 2
        (o % S == 0 && (o + S <= h || o >= h)) ||
            error("extract: slice [$o, $(o + S)) not power-of-2 aligned in $L")
        if o >= h
            v = halve(v, true); o -= h
        else
            v = halve(v, false)
        end
        L = h
    end
    return v
end

"""
Build a combiner region over N carried values (2N scalar block args) ending
in `ret_op` (reduce_return / scan_return). Known single-value combiners map
to arith ops (with signedness from the Julia eltype); arbitrary Julia
functions are compiled as subprograms through cuTile's own front end.
"""
function combiner_region(@nospecialize(combiner), jelts::Vector{DataType},
                         ets::Vector{IR.Type}, ret_op)
    n = length(ets)
    region = IR.Region()
    blk = IR.Block([ets; ets], [IR.Location() for _ in 1:2n])
    push!(region, blk)
    IR.activate(blk)
    try
        xs = [IR.argument(blk, i) for i in 1:n]
        ys = [IR.argument(blk, n + i) for i in 1:n]
        cname = fname(combiner)
        jelt = jelts[1]; et = ets[1]
        isflt = jelt <: AbstractFloat
        isuns = jelt <: Unsigned
        combined = if n == 1 && cname === :+
            [isflt ? v1(arith.addf(xs[1], ys[1]; result=et)) : v1(arith.addi(xs[1], ys[1]; result=et))]
        elseif n == 1 && cname === :max
            [isflt ? v1(arith.maximumf(xs[1], ys[1]; result=et)) :
             isuns ? v1(arith.maxui(xs[1], ys[1]; result=et)) : v1(arith.maxsi(xs[1], ys[1]; result=et))]
        elseif n == 1 && cname === :min
            [isflt ? v1(arith.minimumf(xs[1], ys[1]; result=et)) :
             isuns ? v1(arith.minui(xs[1], ys[1]; result=et)) : v1(arith.minsi(xs[1], ys[1]; result=et))]
        elseif n == 1 && cname === :*
            [isflt ? v1(arith.mulf(xs[1], ys[1]; result=et)) : v1(arith.muli(xs[1], ys[1]; result=et))]
        elseif n == 1 && cname === :|
            [v1(arith.ori(xs[1], ys[1]; result=et))]
        elseif n == 1 && cname === :&
            [v1(arith.andi(xs[1], ys[1]; result=et))]
        elseif n == 1 && (cname === :xor || cname === :⊻)
            [v1(arith.xori(xs[1], ys[1]; result=et))]
        elseif combiner isa Function
            emit_combiner_subprogram(combiner, jelts, xs, ys)
        else
            error("combiner_region: unsupported combiner $combiner")
        end
        ret_op(IR.Value[combined...])
    finally
        IR.deactivate(blk)
    end
    return region
end

function emit_combiner_subprogram(@nospecialize(f), jelts::Vector{DataType},
                                  xs::Vector{IR.Value}, ys::Vector{IR.Value})
    n = length(jelts)
    # cuTile's convention pairs each tile's (a, b) adjacently:
    # f(a_1, b_1, a_2, b_2, ...), with plain scalar types (the front end
    # promotes them to 0-D tiles; Base.isless etc. then inline to intrinsics).
    argt = Tuple{[S for T in jelts for S in (T, T)]...}
    results = ct.code_structured(f, argt; optimize=true)
    length(results) == 1 || error("combiner subprogram: expected one specialization")
    sci, _ = results[1]
    cg = CG()
    flat = Any[]
    for i in 1:n
        push!(flat, xs[i]); push!(flat, ys[i])
    end
    nslots = length(sci.argtypes)
    if nslots == 2 && widenconst(sci.argtypes[2]) <: Tuple
        # vararg method: slot 2 is the slurped args tuple
        cg.args[2] = flat
    elseif nslots == 2n + 1
        for (i, v) in enumerate(flat)
            cg.args[1 + i] = v
        end
    else
        error("combiner subprogram: unexpected arg layout ($(nslots) slots for $(2n) values)")
    end
    walk_block!(cg, sci.entry)
    term = sci.entry.terminator
    term isa ReturnNode || error("combiner subprogram: expected a plain return")
    rv = resolve(cg, term.val)
    vals = rv isa Vector{Any} ? rv : Any[rv]
    return IR.Value[v isa IR.Value ? v : materialize(v, IR.type(xs[min(i, n)]))
                    for (i, v) in enumerate(vals)]
end

# ----------------------------------------------------------------------------
# Control flow
# ----------------------------------------------------------------------------

result_types(@nospecialize(typ)) = begin
    T = widenconst(typ)
    T <: Tuple ? IR.Type[tile_type(p) for p in T.parameters] : IR.Type[tile_type(T)]
end

function region_from_block!(cg::CG, blk::SBlock, res_types::Vector{IR.Type};
                            blockarg_types::Vector{IR.Type}=IR.Type[],
                            blockarg_ids::Vector{Int}=Int[],
                            keep::Union{Nothing,Vector{Bool}}=nothing)
    region = IR.Region()
    b = IR.Block(blockarg_types, [IR.Location() for _ in blockarg_types])
    push!(region, b)
    IR.activate(b)
    try
        for (i, id) in enumerate(blockarg_ids)
            cg.blockargs[id] = IR.argument(b, i)
        end
        walk_block!(cg, blk)
        term = blk.terminator
        if term isa YieldOp || term isa ContinueOp
            tvals = keep === nothing ? term.values : term.values[keep]
            vals = IR.Value[]
            for (i, x) in enumerate(tvals)
                r = resolve(cg, x)
                push!(vals, r isa IR.Value ? asvalue(r) : materialize(r, res_types[i]))
            end
            scf.yield(vals)
        elseif term === nothing
            scf.yield(IR.Value[])
        else
            error("region_from_block!: unhandled terminator $(typeof(term))")
        end
    finally
        IR.deactivate(b)
    end
    return region
end

# The structurizer sometimes carries a token as the singleton instance
# `TokenType()` (a `Core.Const`-like value) rather than the type itself.
is_token_type(@nospecialize(T)) =
    nameof(T isa Type ? T : T isa Core.Const ? widenconst(T) : typeof(T)) === :TokenType

function emit_if!(cg::CG, op::IfOp, @nospecialize(typ))
    T = widenconst(typ)
    # tokens riding in if-results are dropped like loop carries
    params = T === Nothing ? Any[] : T <: Tuple ? collect(T.parameters) : Any[T]
    keep = Bool[!is_token_type(p) for p in params]
    res_types = IR.Type[tile_type(p) for p in params[keep]]
    cond = asvalue(resolve(cg, op.condition))
    then_r = region_from_block!(cg, op.then_region, res_types; keep)
    else_r = region_from_block!(cg, op.else_region, res_types; keep)
    ifop = scf.if_(cond; results=res_types, thenRegion=then_r, elseRegion=else_r)
    out = Any[]
    k = 0
    for kept in keep
        push!(out, kept ? IR.result(ifop, (k += 1)) : nothing)
    end
    return out
end

function emit_for!(cg::CG, op::ForOp, @nospecialize(typ))
    # The induction variable has the range's Julia type, as in the native
    # backend: an Int64 range is an i64 loop. scf.for needs all three bounds
    # of that type.
    iv_jt = widenconst(op.iv_arg.type)
    iv_t = iv_jt <: Integer && sizeof(iv_jt) == 8 ? IR.Type(Int64) : IR.Type(Int32)
    bound(x) = let r = resolve(cg, x)
        r isa IR.Value ? as_int(r, iv_t) : const_int(Int(r), iv_t)
    end
    lb = bound(op.lower); ub = bound(op.upper); st = bound(op.step)

    # Tokens (dropped in this backend) may be loop-carried; filter them out
    # of the scf carries by their inferred types (as emit_if! does for branch
    # results) and bind their block args to `nothing`. Constant inits are
    # materialized against the inferred carry type.
    carry_args = [a for a in op.body.args if a.id != op.iv_arg.id]
    keep = Bool[!is_token_type(a.type) for a in carry_args]
    inits = IR.Value[materialize(resolve(cg, x), tile_type(a.type))
                     for (a, x) in zip(carry_args[keep], op.init_values[keep])]
    init_types = IR.Type[IR.type(v) for v in inits]

    barg_types = IR.Type[iv_t; init_types...]
    barg_ids = Int[op.iv_arg.id; [a.id for a in carry_args[keep]]...]
    for (j, a) in enumerate(carry_args)
        keep[j] || (cg.blockargs[a.id] = nothing)
    end
    region = region_from_block!(cg, op.body, init_types;
                                blockarg_types=barg_types, blockarg_ids=barg_ids, keep)
    forop = scf.for_(lb, ub, st, inits; results=init_types, region=region)
    out = Any[]
    k = 0
    for kept in keep
        push!(out, kept ? IR.result(forop, (k += 1)) : nothing)
    end
    return out
end

# A general loop (LoopOp: exits through `continue`/`break`, possibly inside
# ifs) as an scf.while whose before region runs the body and ends in
# scf.condition(continuing, values): continue and break values both have the
# carries' arity, so they are the next carries or the results.
function emit_loop!(cg::CG, op::LoopOp, @nospecialize(typ))
    keep = Bool[!is_token_type(a.type) for a in op.body.args]
    types = IR.Type[tile_type(a.type) for a in op.body.args[keep]]
    inits = IR.Value[materialize(resolve(cg, x), t)
                     for (x, t) in zip(op.init_values[keep], types)]
    before = IR.Region()
    bblk = IR.Block(types, [IR.Location() for _ in types])
    push!(before, bblk)
    IR.activate(bblk)
    try
        k = 0
        for (j, a) in enumerate(op.body.args)
            cg.blockargs[a.id] = keep[j] ? IR.argument(bblk, (k += 1)) : nothing
        end
        flag, vals = walk_loop_tail!(cg, collect(op.body), 1, op.body.terminator, types, keep)
        scf.condition(flag, vals)
    finally
        IR.deactivate(bblk)
    end
    after = IR.Region()
    ablk = IR.Block(types, [IR.Location() for _ in types])
    push!(after, ablk)
    IR.activate(ablk)
    try
        scf.yield(IR.Value[IR.argument(ablk, i) for i in 1:length(types)])
    finally
        IR.deactivate(ablk)
    end
    wop = scf.while_(inits; results=types, before, after)
    k = 0
    return Any[kept ? IR.result(wop, (k += 1)) : nothing for kept in keep]
end

loop_exits(blk::SBlock) = blk.terminator isa Union{ContinueOp, BreakOp} ||
    any(((_, e),) -> e.stmt isa IfOp && (loop_exits(e.stmt.then_region) ||
                                         loop_exits(e.stmt.else_region)), collect(blk))

# Walk stmts[i:end] of a loop body and then `term`, returning (continuing,
# values). An if that exits in a branch becomes an scf.if yielding them; the
# statements after it run inside each branch that falls through. `after`
# continues the enclosing block when this one ends in a yield instead of an
# exit (a branch whose exit is in a nested if).
function walk_loop_tail!(cg::CG, stmts, i::Int, term, types::Vector{IR.Type}, keep;
                         after=nothing)
    while i <= length(stmts)
        idx, entry = stmts[i]
        st = entry.stmt
        if st isa IfOp && (loop_exits(st.then_region) || loop_exits(st.else_region))
            rest() = walk_loop_tail!(cg, stmts, i + 1, term, types, keep; after)
            branch(blk) = yield_region() do
                t = blk.terminator
                bind() = (t isa YieldOp && !isempty(t.values) &&
                          (cg.env[idx] = Any[resolve(cg, x) for x in t.values]); rest())
                flag, vals = walk_loop_tail!(cg, collect(blk), 1, t, types, keep; after=bind)
                (flag, vals...)
            end
            cond = asvalue(resolve(cg, st.condition))
            ifop = scf.if_(cond; results=IR.Type[IR.Type(Bool); types],
                           thenRegion=branch(st.then_region), elseRegion=branch(st.else_region))
            return IR.result(ifop, 1), IR.Value[IR.result(ifop, j + 1) for j in 1:length(types)]
        end
        cg.env[idx] = walk_stmt!(cg, st, entry.type)
        cg.envtype[idx] = entry.type
        i += 1
    end
    if !(term isa Union{ContinueOp, BreakOp})
        after === nothing &&
            error("walk_loop_tail!: loop body ends in $(typeof(term)), not continue/break")
        return after()
    end
    vals = IR.Value[materialize(resolve(cg, x), t) for (x, t) in zip(term.values[keep], types)]
    return materialize(term isa ContinueOp, IR.Type(Bool)), vals
end

function emit_while!(cg::CG, op::WhileOp, @nospecialize(typ))
    # Token carries are dropped (see emit_for!); the before/after block-arg
    # types type the inits and the scf.condition args respectively.
    keep = Bool[!is_token_type(a.type) for a in op.before.args]
    inits = IR.Value[materialize(resolve(cg, x), tile_type(a.type))
                     for (a, x) in zip(op.before.args[keep], op.init_values[keep])]
    init_types = IR.Type[IR.type(v) for v in inits]

    # before region: carries in, scf.condition(cond, args) out
    before = IR.Region()
    bblk = IR.Block(init_types, [IR.Location() for _ in init_types])
    push!(before, bblk)
    ckeep = Bool[!is_token_type(a.type) for a in op.after.args]
    local res_types
    IR.activate(bblk)
    try
        k = 0
        for (j, a) in enumerate(op.before.args)
            cg.blockargs[a.id] = keep[j] ? IR.argument(bblk, (k += 1)) : nothing
        end
        walk_block!(cg, op.before)
        term = op.before.terminator::ConditionOp
        cond = asvalue(resolve(cg, term.condition))
        cargs = IR.Value[materialize(resolve(cg, x), tile_type(a.type))
                         for (a, x) in zip(op.after.args[ckeep], term.args[ckeep])]
        res_types = IR.Type[IR.type(v) for v in cargs]
        scf.condition(cond, cargs)
    finally
        IR.deactivate(bblk)
    end

    # after region: condition args in, scf.yield(carries) out
    after_ids = Int[a.id for a in op.after.args]
    for (j, id) in enumerate(after_ids)
        ckeep[j] || (cg.blockargs[id] = nothing)
    end
    after = region_from_block!(cg, op.after, init_types;
                               blockarg_types=res_types,
                               blockarg_ids=after_ids[ckeep], keep)
    wop = scf.while_(inits; results=res_types, before=before, after=after)
    out = Any[]
    k = 0
    for kept in ckeep
        push!(out, kept ? IR.result(wop, (k += 1)) : nothing)
    end
    return out
end

# ----------------------------------------------------------------------------
# Kernel entry
# ----------------------------------------------------------------------------

"Struct-typed kernel argument, tracked as an aggregate for getfield."
struct StructVal
    T::DataType
    fields::Vector{Any}
end

# Largest power-of-2 divisor, capped at 16 (the Python frontend's convention
# for tt.divisibility hints feeding the AxisInfo vectorization analysis).
_div16(v) = v <= 1 ? 1 : min(1 << trailing_zeros(Int(v)), 16)

const EMIT_DIV_ATTRS = Ref(true)
const EMIT_CONST_STRIDE = Ref(true)

_divattr(v) = (EMIT_DIV_ATTRS[] && v >= 2) ?
    IR.Attribute(Dict("tt.divisibility" => IR.Attribute(Int32(v)))) :
    IR.Attribute(Dict{String,IR.Attribute}())

# Locate the ArraySpec parameter of a TileArray type. cuTile ≤0.3 stores the
# spec *type* in parameter 3; cuTile 1.0 stores the spec *instance* in
# parameter 4 (parameter 3 became the index type I).
function _spec_type(@nospecialize(T))
    for spec in T.parameters
        S = spec isa DataType ? spec : typeof(spec)
        S <: ct.ArraySpec && return S
    end
    error("TileArray type $T carries no ArraySpec parameter")
end

function _spec_params(@nospecialize(T))
    P = _spec_type(T).parameters
    # (N, Alignment, Contiguous, StrideDivBy, ShapeDivBy)
    return Int(P[2]), P[3]::Bool, P[4], P[5]
end

function arg_mlir_types!(out::Vector{IR.Type}, attrs::Vector{IR.Attribute}, @nospecialize(T))
    i32 = IR.Type(Int32)
    empty_d = IR.Attribute(Dict{String,IR.Attribute}())
    if T <: ct.TileArray
        push!(out, IR.OpaqueType("tt", "ptr<" * string(storage_type(eltype(T))) * ">"))
        align, contig, sdiv, shdiv = _spec_params(T)
        push!(attrs, _divattr(_div16(align)))
        ity = IR.Type(tilearray_indextype(T))
        for d in 1:ndims(T)   # sizes
            push!(out, ity)
            push!(attrs, _divattr(_div16(d <= length(shdiv) ? shdiv[d] : 0)))
        end
        for d in 1:ndims(T)   # strides
            push!(out, ity)
            push!(attrs, _divattr(_div16(d <= length(sdiv) ? sdiv[d] : 0)))
        end
    elseif Base.issingletontype(T)
        # ghost — contributes nothing
    elseif T <: Number || isprimitivetype(T)
        push!(out, scalar_type(T))
        push!(attrs, empty_d)
    elseif isconcretetype(T) && fieldcount(T) >= 0
        for ft in fieldtypes(T)
            arg_mlir_types!(out, attrs, ft)
        end
    else
        error("emit_ttir: unsupported argument type $T")
    end
end

function arg_value(entry, kref::Ref{Int}, @nospecialize(T))
    if T <: ct.TileArray
        N = ndims(T)
        ptr = IR.argument(entry, kref[] += 1)
        sizes = Any[IR.argument(entry, kref[] += 1) for _ in 1:N]
        strides = Any[IR.argument(entry, kref[] += 1) for _ in 1:N]
        # Contiguous ⇒ stride[1] == 1 statically: fold to a constant so
        # AxisInfo can prove pointer contiguity (the equivalent of Python's
        # equal-to-1 argument specialization).
        _, contig, _, _ = _spec_params(T)
        EMIT_CONST_STRIDE[] && contig && (strides[1] = 1)
        return ArgInfo(ptr, sizes, strides)
    elseif Base.issingletontype(T)
        return T.instance
    elseif T <: Number || isprimitivetype(T)
        return IR.argument(entry, kref[] += 1)
    else
        return StructVal(T, Any[arg_value(entry, kref, ft) for ft in fieldtypes(T)])
    end
end

# ct.code_structured(f, argtypes; optimize=true), keeping the
# `@compiler_options` metadata (launch hints) that it drops
function structured_with_meta(@nospecialize(f), @nospecialize(argtypes))
    stripped, const_argtypes = ct.process_const_argtypes(f, argtypes)
    world = Base.get_world_counter()
    mi = ct.lookup_method_instance(f, stripped; world)
    cache = ct.CacheView{ct.CuTileResults}(:cuTile, world)
    ir, rettype = ct.emit_julia(cache, mi; const_argtypes)
    sci, _, kernel_meta = ct.emit_structured(ir, rettype)
    sci = copy(sci)
    ct.run_passes!(sci)
    return sci, kernel_meta
end

"""
    emit_ttir(f, argtypes; name="kernel") -> (ttir_text, Vector{ArgSpec})

Lower a cuTile kernel to Triton IR text plus the flattened parameter spec
the launcher needs.
"""
function emit_ttir(@nospecialize(f), @nospecialize(argtypes); name::String="kernel",
                   use_tma::Bool=true, hints::Union{Bool,Nothing}=nothing,
                   wide_offsets::Bool=false)
    USE_TMA[] = use_tma
    WIDE_OFFSETS[] = wide_offsets
    h = hints !== nothing ? hints : get(ENV, "TRITON_ARG_ATTRS", "1") != "0"
    EMIT_DIV_ATTRS[] = h
    EMIT_CONST_STRIDE[] = hints !== nothing ? hints :
                          get(ENV, "TRITON_CONST_STRIDE", "1") != "0"
    sci, kernel_meta = structured_with_meta(f, argtypes)

    ctx = IR.Context()
    IR.activate(ctx)
    text = ""
    hasdot = false
    hasatomic = false
    spans = Vector{SpanTerm}[]
    argspec = ArgSpec[]
    try
        IR.allow_unregistered_dialects!(true)
        IR.load_all_available_dialects()

        # Build parameter list from sci.argtypes (slot 1 = the function),
        # flattening nested structs recursively like cuTile's _flatten_static!.
        param_types = IR.Type[]
        i32 = IR.Type(Int32)
        arg_attr_dicts = IR.Attribute[]
        for (i, AT) in enumerate(sci.argtypes)
            i == 1 && continue
            AT isa Core.Const && continue
            T = widenconst(AT)
            arg_mlir_types!(param_types, arg_attr_dicts, T)
            push!(argspec, T <: ct.TileArray ? ArgSpec(:tilearray, eltype(T), ndims(T), tilearray_indextype(T)) :
                           T <: Number ? ArgSpec(:scalar, T, 0) : ArgSpec(:other, Nothing, 0))
        end

        # implicit trailing KernelState seed param (mirrors cuTile's ABI)
        push!(param_types, i32)
        push!(arg_attr_dicts, IR.Attribute(Dict{String,IR.Attribute}()))

        mod = IR.Module(IR.Location())
        reg = IR.Region()
        entry = IR.Block(param_types, [IR.Location() for _ in param_types])
        push!(reg, entry)

        cg = CG()
        append!(cg.params, IR.argument(entry, i) for i in 1:length(param_types))
        # Wire params back to Julia slots.
        kref = Ref(0)
        for (i, AT) in enumerate(sci.argtypes)
            i == 1 && continue
            if AT isa Core.Const
                cg.args[i] = AT.val
                continue
            end
            cg.args[i] = arg_value(entry, kref, widenconst(AT))
        end

        cg.seed = IR.argument(entry, length(param_types))

        IR.activate(entry)
        try
            walk_block!(cg, sci.entry; at_top=true)
            tt.return_(IR.Value[])
        finally
            IR.deactivate(entry)
        end

        ftype = IR.FunctionType(param_types, [])
        funcop = tt.func(;
            sym_name=IR.Attribute(name),
            function_type=IR.Attribute(API.mlirTypeAttrGet(ftype)),
            sym_visibility=IR.Attribute("public"),
            arg_attrs=IR.Attribute(arg_attr_dicts),
            body=reg,
        )
        push!(IR.body(mod), funcop)
        text = string(mod)
        hasdot = cg.has_dot
        hasatomic = cg.has_atomic
        spans = cg.spans
    finally
        IR.deactivate(ctx)
    end
    return text, argspec, (; has_dot=hasdot, has_atomic=hasatomic, spans, kernel_meta)
end

end # module
