# TritonEmitter: lower cuTile StructuredIRCode to Triton IR (tt dialect) text.
#
#   Julia kernel + argtypes
#     → ct.code_structured(f, argtypes)     (cuTile front end, target-agnostic)
#     → emit_ttir(sci, argtypes)            (this file: MLIR.jl, unregistered tt)
#     → triton wheel IRSource compile       (compile_ttir.py)
#     → CUDA.jl launch                      (TritonRun.jl)
#
# Kernel ABI (must match TritonRun.flatten_args): each TileArray{T,N} becomes
# (ptr::!tt.ptr<T>, sizes::N×i32, strides::N×i32) in Julia dim order; plain
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

# Upstream bugfix (candidate PR for cuTile.jl): promote_scalar_type does
# `T.parameters` after `T <: Tuple`, but a Union of tuples passes that subtype
# check and has no `.parameters` field → FieldError inside the front end on
# Julia 1.12 (hvcat/hvncat kernels). Guard Unions conservatively. This
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
    padding::Symbol            # :undetermined | :zero | :neginf
    contiguous::Bool           # ArraySpec.Contiguous (Julia dim 1 has stride 1)
    desc::Union{Nothing,IR.Value}  # !tt.tensordesc — set when the TMA path is used
    order::Union{Nothing,Vector{Int}}  # dim permutation: tile dim d ↔ array dim order[d]
end

# Kernel-creation-time switch for the TMA/descriptor lowering (per-view
# legality still gates it; this is the opt-out).
const USE_TMA = Ref(true)

# TMA legality for a partition view: 2-D, unit-stride inner dim, box dims
# within CUtensorMap's 256-element limit, ≥16-byte inner box (hardware
# minimum — a 4-byte inner box traps the GPU). OOB zero-fill covers :zero
# and :undetermined directly; :neginf is recovered with an arithmetic select
# after the load (see emit_view_load). Dtype heuristic from measurement:
# descriptors are a large win for ≤16-bit element types and a regression for
# f32/tf32 on triton 3.7's Hopper pipeline, so gate on element size.
tma_eligible(N, tile_shape, pad, contiguous, elty) =
    USE_TMA[] && 2 <= N <= 5 && contiguous &&
    sizeof(elty) <= 2 &&
    all(s -> 1 <= s <= 256, tile_shape) &&
    tile_shape[1] * sizeof(elty) >= 16

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
        push!(shape_vals, s isa IR.Value ? s : const_i32(Int(s)))
        if d == 1
            push!(stride_vals, const_scalar(1, i64))
        else
            st = view.strides[d]
            stv = st isa IR.Value ? st : const_i32(Int(st))
            push!(stride_vals, v1(arith.extsi(stv; out=i64)))
        end
    end
    rev_sh = reverse(view.tile_shape)
    desc_t = IR.OpaqueType("tt", "tensordesc<tensor<" *
                           join(rev_sh, "x") * "x" * string(et) * ">>")
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
end
CG() = CG(Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), Dict{Int,Any}(), nothing, false, false)

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
        return IR.OpaqueType("tt", "ptr<" * string(scalar_type(eltype(T))) * ">")
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
A linear (single) tile index over an N-D partition view delinearizes
column-major over the per-dim tile counts: idx_d = (lin ÷ ∏nt_{<d}) mod nt_d.
"""
function delinearize_index(view::ViewInfo, idxs::Vector{Any})
    N = length(view.tile_shape)
    (length(idxs) == 1 && N > 1) || return idxs
    i32 = IR.Type(Int32)
    lin = idxs[1] isa IR.Value ? as_i32(idxs[1]) : const_i32(Int(idxs[1]))
    out = Any[]
    for d in 1:N
        size_d = view.sizes[d]
        size_v = size_d isa IR.Value ? size_d : const_i32(Int(size_d))
        nt = v1(arith.ceildivsi(size_v, const_i32(view.tile_shape[d]); result=i32))
        if d < N
            push!(out, v1(arith.remsi(lin, nt; result=i32)))
            lin = v1(arith.divsi(lin, nt; result=i32))
        else
            push!(out, lin)
        end
    end
    return out
end

"""
Compute (ptrs, mask) for a partition-view access at tile index `idxs`
(0-based, per Julia dim). Returns (ptrs::Value, mask::Value or nothing).
"""
function view_ptrs_mask(view::ViewInfo, idxs::Vector{Any}, cg::CG)
    idxs = delinearize_index(view, idxs)
    sh = view.tile_shape
    N = length(sh)
    i32 = IR.Type(Int32)
    et = scalar_type(view.elty)
    ptr_t = IR.OpaqueType("tt", "ptr<" * string(et) * ">")

    flat_offs = nothing
    mask = nothing
    for d in 1:N
        # order: tile dim d walks array dim ad (identity when unset)
        ad = view.order === nothing ? d : view.order[d]
        S = sh[d]
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_i32(idx) : const_i32(Int(idx))
        start = v1(arith.muli(idx_v, const_i32(S); result=i32))
        rng = v1(tt.make_range(; result=IR.TensorType([S], i32), start=Int32(0), end_=Int32(S)))
        offs1 = v1(arith.addi(v1(tt.splat(start; result=IR.TensorType([S], i32))), rng;
                              result=IR.TensorType([S], i32)))
        offs_f = expand_to_full(offs1, sh, d, i32)
        full_t = IR.type(offs_f)

        # bounds mask along this dim: offs < size_ad
        size_d = view.sizes[ad]
        size_v = size_d isa IR.Value ? size_d : const_i32(Int(size_d))
        size_f = v1(tt.splat(size_v; result=full_t))
        m = v1(arith.cmpi(offs_f, size_f; predicate=IR.Attribute(Int64(2)), # slt
                          result=IR.TensorType(reverse(sh), IR.Type(Bool))))
        mask = mask === nothing ? m : v1(arith.andi(mask, m; result=IR.type(m)))

        # pointer offset contribution: offs * stride_ad (skip for unit stride
        # so AxisInfo sees contiguity directly)
        stride_d = view.strides[ad]
        contrib = if stride_d isa Int && stride_d == 1
            offs_f
        else
            stride_v = stride_d isa IR.Value ? stride_d : const_i32(Int(stride_d))
            stride_f = v1(tt.splat(stride_v; result=full_t))
            v1(arith.muli(offs_f, stride_f; result=full_t))
        end
        flat_offs = flat_offs === nothing ? contrib :
                    v1(arith.addi(flat_offs, contrib; result=full_t))
    end

    ptrs_t = IR.TensorType(reverse(sh), ptr_t)
    ptrs0 = v1(tt.splat(view.ptr; result=ptrs_t))
    ptrs = v1(tt.addptr(ptrs0, flat_offs; result=ptrs_t))
    return ptrs, mask
end

# Element offsets for descriptor access: idx_d * tile_d, reversed axis order.
function desc_offsets(view::ViewInfo, idxs::Vector{Any})
    idxs = delinearize_index(view, idxs)
    i32 = IR.Type(Int32)
    offs = IR.Value[]
    for d in length(view.tile_shape):-1:1
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_i32(idx) : const_i32(Int(idx))
        push!(offs, v1(arith.muli(idx_v, const_i32(view.tile_shape[d]); result=i32)))
    end
    return offs
end

"Just the in-bounds mask of a partition-view access (no pointer math)."
function view_bounds_mask(view::ViewInfo, idxs::Vector{Any})
    idxs = delinearize_index(view, idxs)
    sh = view.tile_shape
    N = length(sh)
    i32 = IR.Type(Int32)
    mask = nothing
    for d in 1:N
        ad = view.order === nothing ? d : view.order[d]
        S = sh[d]
        idx = idxs[d]
        idx_v = idx isa IR.Value ? as_i32(idx) : const_i32(Int(idx))
        start = v1(arith.muli(idx_v, const_i32(S); result=i32))
        rng = v1(tt.make_range(; result=IR.TensorType([S], i32), start=Int32(0), end_=Int32(S)))
        offs1 = v1(arith.addi(v1(tt.splat(start; result=IR.TensorType([S], i32))), rng;
                              result=IR.TensorType([S], i32)))
        offs_f = expand_to_full(offs1, sh, d, i32)
        size_d = view.sizes[ad]
        size_v = size_d isa IR.Value ? size_d : const_i32(Int(size_d))
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
        if view.padding === :neginf
            # TMA can only zero-fill OOB; recover NegInf semantics with an
            # arithmetic select — data movement stays on TMA, the -inf mask
            # is register-only index math (what Triton attention kernels do).
            m = view_bounds_mask(view, idxs)
            ninf = splat_const(view.tile_shape, -Inf, scalar_type(view.elty))
            return v1(arith.select(m, loaded, ninf; result=res_t))
        end
        return loaded
    end
    ptrs, mask = view_ptrs_mask(view, idxs, cg)
    res_t = tensor_of(view.tile_shape, scalar_type(view.elty))
    if view.padding === :zero
        other = splat_const(view.tile_shape, zero(view.elty <: Integer ? Int : Float64), scalar_type(view.elty))
        return v1(tt.load(ptrs, mask; other=other, result=res_t))
    elseif view.padding === :neginf
        other = splat_const(view.tile_shape, -Inf, scalar_type(view.elty))
        return v1(tt.load(ptrs, mask; other=other, result=res_t))
    else
        return v1(tt.load(ptrs, mask; result=res_t))
    end
end

function emit_view_store(view::ViewInfo, val::IR.Value, idxs::Vector{Any}, cg::CG)
    if view.desc !== nothing
        tt.descriptor_store(view.desc, val, desc_offsets(view, idxs))
        return nothing
    end
    ptrs, mask = view_ptrs_mask(view, idxs, cg)
    tt.store(ptrs, val, mask)
    return nothing
end

# ----------------------------------------------------------------------------
# Predicates / enums
# ----------------------------------------------------------------------------

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
        # cuTile front end marks calls with no Tile IR equivalent; surface the
        # same message at compile time.
        error(stmt.message)
    end
    stmt isa LoopOp &&
        error("walk_stmt!: LoopOp not yet supported by the Triton emitter")
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
                     :fabs => math.absf, :absf => math.absf,
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
        pm = Symbol(resolve(cg, args[3]))
        pad = pm === :Zero ? :zero : pm === :NegInf ? :neginf : :undetermined
        ord = length(args) >= 4 ? resolve(cg, args[4]) : nothing
        order = ord isa Tuple ? Int[ord...] : nothing
        view = ViewInfo(base.ptr, base.sizes, base.strides, base.elty,
                        Int[shape...], pad, base.contiguous, nothing, order)
        N = length(view.tile_shape)
        if order === nothing && tma_eligible(N, view.tile_shape, pad, view.contiguous, view.elty)
            view.desc = emit_descriptor(view)
        end
        return view
    elseif f === :get_index_space_shape
        view = resolve(cg, args[1])::ViewInfo
        d = Int(resolve(cg, args[2])) + 1   # intrinsic axis is 0-based Julia dim
        size_d = view.sizes[d]
        size_v = size_d isa IR.Value ? size_d : const_i32(Int(size_d))
        return v1(arith.ceildivsi(size_v, const_i32(view.tile_shape[d]); result=IR.Type(Int32)))
    elseif f === :load_partition_view
        view = resolve(cg, args[1])::ViewInfo
        idxs = resolve(cg, args[4])
        return emit_view_load(view, Any[idxs...], cg)
    elseif f === :store_partition_view
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
            return v1(tt.load(ptrs, mv; other=other, result=res_t))
        end
        return mv === nothing ? v1(tt.load(ptrs; result=res_t)) :
                                v1(tt.load(ptrs, mv; result=res_t))
    elseif f === :store_ptr_tko
        # scatter: (ptrs, values, latency, mask[, token])
        ptrs = asvalue(resolve(cg, args[1]))
        val = asvalue(resolve(cg, args[2]))
        mask = length(args) >= 4 ? resolve(cg, args[4]) : nothing
        mv = mask isa IR.Value ? mask : nothing
        mv === nothing ? tt.store(ptrs, val) : tt.store(ptrs, val, mv)
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
    elseif haskey(FLT_BIN, f)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(FLT_BIN[f](av, bv; result=IR.type(av)))
    elseif f === :divi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(arith.divsi(av, bv; result=IR.type(av)))
    elseif f === :cldi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        op2 = Symbol(resolve(cg, args[3])) === :Signed ? arith.ceildivsi : arith.ceildivui
        return v1(op2(av, bv; result=IR.type(av)))
    elseif f === :fldi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        op2 = Symbol(resolve(cg, args[3])) === :Signed ? arith.floordivsi : arith.divui
        return v1(op2(av, bv; result=IR.type(av)))
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
        return vals[1]
    elseif f === :remi
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        return v1(arith.remsi(av, bv; result=IR.type(av)))
    elseif f === :negf
        # triton doesn't legalize arith.negf; match Python's `0 - x`
        a = asvalue(resolve(cg, args[1]))
        return v1(arith.subf(materialize(0, IR.type(a)), a; result=IR.type(a)))
    elseif f === :negi
        a = asvalue(resolve(cg, args[1]))
        return v1(arith.subi(materialize(0, IR.type(a)), a; result=IR.type(a)))
    elseif f === :shri
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        signed = Symbol(resolve(cg, args[3])) === :Signed
        op = signed ? arith.shrsi : arith.shrui
        return v1(op(av, bv; result=IR.type(av)))
    elseif f === :pow
        # triton's pipeline has no math.powf legalization; use exp2(y·log2 x)
        a = resolve(cg, args[1]); b = resolve(cg, args[2])
        av, bv = _pair_values(a, b)
        lg = v1(math.log2(av; result=IR.type(av)))
        pr = v1(arith.mulf(bv, lg; result=IR.type(av)))
        return v1(math.exp2(pr; result=IR.type(av)))
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
        mv = mask isa IR.Value ? mask : nothing
        RT = widenconst(typ)
        jelt = tile_eltype(RT)
        isflt = jelt <: AbstractFloat
        isuns = jelt <: Unsigned
        rmw = f === :atomic_add ? (isflt ? 5 : 4) :          # FADD : ADD
              f === :atomic_max ? (isuns ? 8 : 6) :          # UMAX : MAX
              f === :atomic_min ? (isuns ? 9 : 7) :          # UMIN : MIN
              f === :atomic_and ? 1 : f === :atomic_or ? 2 :
              f === :atomic_xor ? 3 : 10                     # XCHG
        return v1(tt.atomic_rmw(ptrs, val, mv; result=tile_type(typ),
                                atomic_rmw_op=IR.Attribute(Int32(rmw)),
                                sem=IR.Attribute(Int32(mem_sem(resolve(cg, args[4])))),
                                scope=IR.Attribute(Int32(mem_scope(resolve(cg, args[5]))))))
    elseif f === :atomic_cas
        # (ptr_tile, expected, desired, mask, memory_order, memory_scope[, token])
        cg.has_atomic = true
        mask = resolve(cg, args[4])
        mask isa IR.Value &&
            error("atomic_cas: masked CAS has no tt.atomic_cas equivalent yet")
        ptrs = asvalue(resolve(cg, args[1]))
        cmp = materialize(resolve(cg, args[2]), tile_type(typ))
        val = materialize(resolve(cg, args[3]), tile_type(typ))
        return v1(tt.atomic_cas(ptrs, cmp, val; result=tile_type(typ),
                                sem=IR.Attribute(Int32(mem_sem(resolve(cg, args[5])))),
                                scope=IR.Attribute(Int32(mem_scope(resolve(cg, args[6]))))))

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
        tt.assert(cv; message=IR.Attribute(String(msg)))
        return nothing
    elseif f === :MakeTokenNode || f === :JoinTokensNode
        return nothing
    elseif f === :fpmode_begin || f === :fpmode_end
        # fast-math mode scoping: Triton has no region-scoped fp modes; ops
        # default to its standard fast-math behavior. Accept and ignore.
        return nothing
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

is_token_type(@nospecialize(T)) = nameof(widenconst(T)) === :TokenType

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
    i32 = IR.Type(Int32)
    lb = let r = resolve(cg, op.lower); r isa IR.Value ? r : const_i32(Int(r)) end
    ub = let r = resolve(cg, op.upper); r isa IR.Value ? r : const_i32(Int(r)) end
    st = let r = resolve(cg, op.step); r isa IR.Value ? r : const_i32(Int(r)) end

    # Tokens (dropped in this backend) may be loop-carried; filter them out
    # of the scf carries and bind their block args to `nothing`.
    rinit = Any[resolve(cg, x) for x in op.init_values]
    keep = Bool[r isa IR.Value || r isa TF32Val for r in rinit]
    inits = IR.Value[asvalue(r) for r in rinit[keep]]
    init_types = IR.Type[IR.type(v) for v in inits]

    carry_ids = Int[a.id for a in op.body.args if a.id != op.iv_arg.id]
    barg_types = IR.Type[i32; init_types...]
    barg_ids = Int[op.iv_arg.id; carry_ids[keep]...]
    for (j, id) in enumerate(carry_ids)
        keep[j] || (cg.blockargs[id] = nothing)
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

function emit_while!(cg::CG, op::WhileOp, @nospecialize(typ))
    # Token carries are dropped (see emit_for!).
    rinit = Any[resolve(cg, x) for x in op.init_values]
    keep = Bool[r isa IR.Value || r isa TF32Val for r in rinit]
    inits = IR.Value[asvalue(r) for r in rinit[keep]]
    init_types = IR.Type[IR.type(v) for v in inits]

    # before region: carries in, scf.condition(cond, args) out
    before = IR.Region()
    bblk = IR.Block(init_types, [IR.Location() for _ in init_types])
    push!(before, bblk)
    local res_types, ckeep
    IR.activate(bblk)
    try
        k = 0
        for (j, a) in enumerate(op.before.args)
            cg.blockargs[a.id] = keep[j] ? IR.argument(bblk, (k += 1)) : nothing
        end
        walk_block!(cg, op.before)
        term = op.before.terminator::ConditionOp
        cond = asvalue(resolve(cg, term.condition))
        rargs = Any[resolve(cg, x) for x in term.args]
        ckeep = Bool[r isa IR.Value || r isa TF32Val for r in rargs]
        cargs = IR.Value[asvalue(r) for r in rargs[ckeep]]
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
        push!(out, IR.OpaqueType("tt", "ptr<" * string(scalar_type(eltype(T))) * ">"))
        align, contig, sdiv, shdiv = _spec_params(T)
        push!(attrs, _divattr(_div16(align)))
        for d in 1:ndims(T)   # sizes
            push!(out, i32)
            push!(attrs, _divattr(_div16(d <= length(shdiv) ? shdiv[d] : 0)))
        end
        for d in 1:ndims(T)   # strides
            push!(out, i32)
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

"""
    emit_ttir(f, argtypes; name="kernel") -> (ttir_text, Vector{ArgSpec})

Lower a cuTile kernel to Triton IR text plus the flattened parameter spec
the launcher needs.
"""
function emit_ttir(@nospecialize(f), @nospecialize(argtypes); name::String="kernel",
                   use_tma::Bool=true, hints::Union{Bool,Nothing}=nothing)
    USE_TMA[] = use_tma
    h = hints !== nothing ? hints : get(ENV, "TRITON_ARG_ATTRS", "1") != "0"
    EMIT_DIV_ATTRS[] = h
    EMIT_CONST_STRIDE[] = hints !== nothing ? hints :
                          get(ENV, "TRITON_CONST_STRIDE", "1") != "0"
    results = ct.code_structured(f, argtypes; optimize=true)
    length(results) == 1 || error("expected a single specialization")
    sci, _ = results[1]

    ctx = IR.Context()
    IR.activate(ctx)
    text = ""
    hasdot = false
    hasatomic = false
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
            push!(argspec, T <: ct.TileArray ? ArgSpec(:tilearray, eltype(T), ndims(T)) :
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
    finally
        IR.deactivate(ctx)
    end
    return text, argspec, (; has_dot=hasdot, has_atomic=hasatomic)
end

end # module
