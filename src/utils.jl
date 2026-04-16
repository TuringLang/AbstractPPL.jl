module Utils

# Vectorisation utilities

# This utility only supports a small structural subset so flattening stays
# predictable and reconstruction can use `x` as the template.

flat_length(x::Union{Real,Complex}) = 1
flat_length(x::AbstractArray{<:Union{Real,Complex}}) = length(x)
flat_length(x::Tuple) = mapreduce(flat_length, +, x; init=0)
flat_length(x::NamedTuple) = mapreduce(flat_length, +, values(x); init=0)
flat_length(x) = throw(ArgumentError("This value cannot be flattened into a vector."))

flat_eltype(x::Union{Real,Complex}) = typeof(x)
flat_eltype(x::AbstractArray{T}) where {T<:Union{Real,Complex}} = T
flat_eltype(x::Tuple) = mapreduce(flat_eltype, promote_type, x; init=Float64)
flat_eltype(x::NamedTuple) = mapreduce(flat_eltype, promote_type, values(x); init=Float64)
flat_eltype(x) = throw(ArgumentError("This value cannot be flattened into a vector."))

"""
    flatten_to!!(buf, x)

Flatten `x` into the vector-like buffer `buf`.

Supported `x` values are:
- `Real`
- `Complex`
- `AbstractArray{<:Union{Real,Complex}}`
- `Tuple` recursively containing supported values
- `NamedTuple` recursively containing supported values

Pass `nothing` as `buf` to allocate a new vector.
"""
function flatten_to!!(::Nothing, x)
    buf = Vector{flat_eltype(x)}(undef, flat_length(x))
    return flatten_to!!(buf, x)
end

function flatten_to!!(buf::AbstractVector, x)
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
    buf::AbstractVector, x::AbstractArray{<:Union{Real,Complex}}, offset::Int
)
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

function _flatten_to!(buf::AbstractVector, x, offset::Int)
    throw(ArgumentError("This value cannot be flattened into a vector."))
end

function unflatten(x, buf::AbstractVector)
    n = flat_length(x)
    length(buf) == n || throw(
        DimensionMismatch("Expected a vector of length $n, but got length $(length(buf))."),
    )
    value, offset = _unflatten(x, buf, 1)
    offset == length(buf) + 1 || throw(
        DimensionMismatch("Expected a vector of length $n, but got length $(length(buf))."),
    )
    return value
end

function _unflatten(x::Union{Real,Complex}, buf::AbstractVector, offset::Int)
    return buf[offset], offset + 1
end

function _unflatten(
    x::AbstractArray{<:Union{Real,Complex}}, buf::AbstractVector, offset::Int
)
    n = length(x)
    value = similar(x, promote_type(eltype(x), eltype(buf)))
    copyto!(value, 1, buf, offset, n)
    return value, offset + n
end

_unflatten(::Tuple{}, buf::AbstractVector, offset::Int) = (), offset

function _unflatten(x::Tuple, buf::AbstractVector, offset::Int)
    first_value, offset = _unflatten(first(x), buf, offset)
    rest_value, offset = _unflatten(Base.tail(x), buf, offset)
    return (first_value, rest_value...), offset
end

function _unflatten(::NamedTuple{(),Tuple{}}, buf::AbstractVector, offset::Int)
    NamedTuple(), offset
end

function _unflatten(x::NamedTuple{Names}, buf::AbstractVector, offset::Int) where {Names}
    first_name = first(Names)
    first_value, offset = _unflatten(getfield(x, first_name), buf, offset)
    rest_names = Base.tail(Names)
    rest_values = Base.tail(values(x))
    rest_value, offset = _unflatten(NamedTuple{rest_names}(rest_values), buf, offset)
    return merge(NamedTuple{(first_name,)}((first_value,)), rest_value), offset
end

"""
    unflatten_to!!(x, buf)

Reconstruct a value from the vector-like buffer `buf` using `x` as the structural template.

Supported `x` values are:
- `Real`
- `Complex`
- `AbstractArray{<:Union{Real,Complex}}`
- `Tuple` recursively containing supported values
- `NamedTuple` recursively containing supported values
"""
unflatten_to!!(x, buf::AbstractVector) = unflatten(x, buf)

end # module
