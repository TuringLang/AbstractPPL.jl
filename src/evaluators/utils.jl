# Vectorisation utilities

# Opt-in: `Array` is the only `AbstractArray` subtype that round-trips cleanly
# through `similar` + `copyto!` while preserving structure. `SubArray` is
# excluded because `similar(::SubArray)` returns a plain `Array`, so the view
# wrapper is silently dropped on `unflatten_to!!` and `typeof(x2) == typeof(x)`
# would not hold. Structured wrappers (`Symmetric`, `Diagonal`, …), lazy
# wrappers (`Adjoint`/`Transpose`), `OffsetArray`, and custom array types fall
# through to the catch-all rejection — callers must `collect` first.
#
# TODO: extend with proper support for structured arrays (independent-entry
# packing) and factorisation types (Cholesky in particular is needed for PPL
# covariance parameters).
const FlattableArray{T} = Array{T}

flat_length(x::Union{Real,Complex}) = 1
flat_length(x::FlattableArray{<:Union{Real,Complex}}) = length(x)
flat_length(::Tuple{}) = 0
flat_length(x::Tuple) = sum(flat_length, x)
flat_length(::NamedTuple{(),Tuple{}}) = 0
flat_length(x::NamedTuple) = sum(flat_length, values(x))
flat_length(x) = throw(ArgumentError("This value cannot be flattened into a vector."))

flat_eltype(x::Union{Real,Complex}) = typeof(x)
flat_eltype(x::FlattableArray{T}) where {T<:Union{Real,Complex}} = T
flat_eltype(::Tuple{}) = Float64
flat_eltype(x::Tuple) = mapreduce(flat_eltype, promote_type, x)
flat_eltype(::NamedTuple{(),Tuple{}}) = Float64
flat_eltype(x::NamedTuple) = mapreduce(flat_eltype, promote_type, values(x))
flat_eltype(x) = throw(ArgumentError("This value cannot be flattened into a vector."))

"""
    flatten_to!!(buf, x)

Flatten `x` into the vector-like buffer `buf`.

Supported `x` values are:
- `Real`
- `Complex`
- `Array{<:Union{Real,Complex}}` (other `AbstractArray` subtypes, including
  views, must be `collect`ed first)
- `Tuple` recursively containing supported values
- `NamedTuple` recursively containing supported values

Pass `nothing` as `buf` to allocate a new vector.
"""
function flatten_to!!(::Nothing, x)
    buf = Vector{flat_eltype(x)}(undef, flat_length(x))
    _flatten_to!(buf, x, 1)
    return buf
end

function flatten_to!!(buf::AbstractVector, x)
    Base.require_one_based_indexing(buf)
    n = flat_length(x)
    length(buf) == n || throw(
        DimensionMismatch("Expected a vector of length $n, but got length $(length(buf))."),
    )
    _flatten_to!(buf, x, 1)
    return buf
end

function _flatten_to!(buf::AbstractVector, x::Union{Real,Complex}, offset::Int)
    buf[offset] = x
    return offset + 1
end

function _flatten_to!(
    buf::AbstractVector, x::FlattableArray{<:Union{Real,Complex}}, offset::Int
)
    Base.require_one_based_indexing(x)
    n = length(x)
    copyto!(buf, offset, x, 1, n)
    return offset + n
end

function _flatten_to!(buf::AbstractVector, x::Tuple, offset::Int)
    for value in x
        offset = _flatten_to!(buf, value, offset)
    end
    return offset
end

function _flatten_to!(buf::AbstractVector, x::NamedTuple, offset::Int)
    for value in values(x)
        offset = _flatten_to!(buf, value, offset)
    end
    return offset
end

function _flatten_to!(buf::AbstractVector, x, ::Int)
    throw(ArgumentError("This value cannot be flattened into a vector."))
end

function _unflatten(x::Union{Real,Complex}, buf::AbstractVector, offset::Int)
    return convert(typeof(x), buf[offset]), offset + 1
end

function _unflatten(
    x::FlattableArray{<:Union{Real,Complex}}, buf::AbstractVector, offset::Int
)
    Base.require_one_based_indexing(x)
    n = length(x)
    value = similar(x)
    copyto!(value, 1, buf, offset, n)
    return value, offset + n
end

_unflatten(::Tuple{}, buf::AbstractVector, offset::Int) = (), offset

function _unflatten(x::Tuple, buf::AbstractVector, offset::Int)
    first_value, offset = _unflatten(first(x), buf, offset)
    rest_value, offset = _unflatten(Base.tail(x), buf, offset)
    return (first_value, rest_value...), offset
end

# Generated to keep the result `NamedTuple` type inferable: a recursive `merge`
# over `Base.tail(Names)` erases parameters and breaks `@inferred` callers.
@generated function _unflatten(
    x::NamedTuple{Names}, buf::AbstractVector, offset::Int
) where {Names}
    if isempty(Names)
        return :((NamedTuple(), offset))
    end
    block = Expr(:block, :(off = offset))
    val_syms = Symbol[]
    for name in Names
        v = gensym(name)
        push!(val_syms, v)
        push!(block.args, :(($v, off) = _unflatten(x[$(QuoteNode(name))], buf, off)))
    end
    push!(block.args, :(return (NamedTuple{$Names}(($(val_syms...),)), off)))
    return block
end

"""
    unflatten_to!!(x, buf; check_eltype::Bool=false)

Reconstruct a value from the vector-like buffer `buf` using `x` as the structural template.

Supported `x` values are:
- `Real`
- `Complex`
- `Array{<:Union{Real,Complex}}` (other `AbstractArray` subtypes, including
  views, must be `collect`ed first)
- `Tuple` recursively containing supported values
- `NamedTuple` recursively containing supported values

Pass `check_eltype=true` to emit a warning when `eltype(buf)` differs from
`flat_eltype(x)` (off by default to keep hot paths quiet).
"""
# Always allocates: `_unflatten` calls `similar` for each array field. Gains from
# buffer reuse are negligible relative to gradient computation cost.
#
# Heterogeneous round-trip: the flat buffer widens, but leaves are rebuilt
# from `x`'s types, so `typeof(x2) == typeof(x)`. E.g.
#
#     x  = (1.0, [2.0, 3.0], (4.0 + 1.0im,))      # buffer widens to ComplexF64
#     x2 = unflatten_to!!(x, flatten_to!!(nothing, x))
#     # x2 == (1.0, [2.0, 3.0], (4.0 + 1.0im,))
#     # x2 == x      → true
#     # typeof(x2) == typeof(x) → true
function unflatten_to!!(x, buf::AbstractVector; check_eltype::Bool=false)
    Base.require_one_based_indexing(buf)
    n = flat_length(x)
    length(buf) == n || throw(
        DimensionMismatch("Expected a vector of length $n, but got length $(length(buf))."),
    )
    if check_eltype
        expected = flat_eltype(x)
        eltype(buf) === expected || @warn(
            "Buffer eltype `$(eltype(buf))` differs from `flat_eltype(x) = $expected`; reconstructing using the leaf types from `x`. An `InexactError` will be thrown if any value in `buf` cannot be converted back to the corresponding leaf type."
        )
    end
    value, _ = _unflatten(x, buf, 1)
    return value
end
