using ADTypes: ADTypes

"""
    DerivativeOrder{K}

Trait indicating the maximum derivative order supported by a prepared evaluator.
`K` must be 0, 1, or 2.
"""
struct DerivativeOrder{K}
    function DerivativeOrder{K}() where {K}
        K isa Int && 0 <= K <= 2 ||
            throw(ArgumentError("DerivativeOrder parameter must be 0, 1, or 2, got $K"))
        return new{K}()
    end
end

Base.isless(::DerivativeOrder{K}, ::DerivativeOrder{L}) where {K,L} = K < L

"""
    capabilities(T::Type)
    capabilities(x)

Return the [`DerivativeOrder`](@ref) supported by a prepared evaluator type or instance.
Returns `nothing` unless a prepared evaluator type defines support explicitly.
"""
capabilities(::Type) = nothing
capabilities(x) = capabilities(typeof(x))

"""
    prepare(problem, values::NamedTuple)
    prepare(adtype::ADTypes.AbstractADType, problem, values::NamedTuple)

Prepare a callable evaluator for `problem` using `values` to define the input
structure. The two-argument form performs structural preparation only. The
three-argument form additionally sets up AD for [`value_and_gradient`](@ref).
"""
function prepare end

"""
    value_and_gradient(prepared, values::NamedTuple)

Return `(value, gradient)` for runtime inputs consistent with the prepare-time
prototype. Vector-backed evaluators additionally require matching array sizes,
and `gradient` is returned with the prepare-time named structure.
Requires `capabilities(prepared) >= DerivativeOrder{1}()`.
"""
function value_and_gradient end

_check_runtime_compatibility(::Real, ::Real) = nothing
function _check_runtime_compatibility(
    prototype::AbstractArray{<:Real}, value::AbstractArray{<:Real}
)
    typeof(value) === typeof(prototype) ||
        throw(MethodError(_check_runtime_compatibility, (prototype, value)))
    size(value) == size(prototype) || throw(
        DimensionMismatch("got array of size $(size(value)), expected $(size(prototype))"),
    )
    return nothing
end
function _check_runtime_compatibility(
    prototype::NamedTuple{K}, value::NamedTuple{K}
) where {K}
    for (prototype_value, runtime_value) in zip(values(prototype), values(value))
        _check_runtime_compatibility(prototype_value, runtime_value)
    end
    return nothing
end
function _check_runtime_compatibility(prototype, value)
    throw(MethodError(_check_runtime_compatibility, (prototype, value)))
end

function check_runtime_compatibility(prototype::NamedTuple, value::NamedTuple)
    _check_runtime_compatibility(prototype, value)
    return value
end

function check_runtime_type(prototype::NamedTuple, value::NamedTuple)
    typeof(value) === typeof(prototype) ||
        throw(MethodError(check_runtime_type, (prototype, value)))
    return value
end

function flatten_to_vec(prototype::NamedTuple, value::NamedTuple)
    check_runtime_compatibility(prototype, value)
    return flatten_to_vec(value)
end

function unflatten_from_vec(prototype::NamedTuple, value::NamedTuple, vec::AbstractVector)
    check_runtime_compatibility(prototype, value)
    return unflatten_from_vec(prototype, vec)
end

"""
    dimension(prepared)::Int

Return the number of scalar dimensions in the vector view of a prepared evaluator.
"""
function dimension end

_scalar_count(::Real) = 1
_scalar_count(x::AbstractArray{<:Real}) = length(x)
function _scalar_count(nt::NamedTuple)
    n = 0
    for v in values(nt)
        n += _scalar_count(v)
    end
    return n
end

_vec_eltype(x::Real) = typeof(x)
_vec_eltype(x::AbstractArray{T}) where {T<:Real} = T
function _vec_eltype(nt::NamedTuple)
    isempty(nt) && return Float64
    return mapreduce(_vec_eltype, promote_type, values(nt))
end

function _flatten!(vec::AbstractVector, s::Real, offset::Int)
    vec[offset] = s
    return offset + 1
end
function _flatten!(vec::AbstractVector, a::AbstractArray{<:Real}, offset::Int)
    n = length(a)
    copyto!(vec, offset, a, 1, n)
    return offset + n
end
function _flatten!(vec::AbstractVector, nt::NamedTuple, offset::Int)
    for v in values(nt)
        offset = _flatten!(vec, v, offset)
    end
    return offset
end

function flatten_to_vec(nt::NamedTuple)
    n = _scalar_count(nt)
    vec = Vector{_vec_eltype(nt)}(undef, n)
    _flatten!(vec, nt, 1)
    return vec
end

function _unflatten(::Real, vec::AbstractVector, offset::Int)
    offset <= length(vec) || throw(BoundsError(vec, offset))
    return vec[offset], offset + 1
end
function _unflatten(values::AbstractArray{<:Real}, vec::AbstractVector, offset::Int)
    n = length(values)
    offset + n - 1 <= length(vec) || throw(BoundsError(vec, offset:(offset + n - 1)))
    result = reshape(@view(vec[offset:(offset + n - 1)]), size(values))
    return result, offset + n
end
# Recursive peel keeps this type-stable (Enzyme needs it).
function _unflatten(::NamedTuple{(),Tuple{}}, ::AbstractVector, offset::Int)
    return NamedTuple(), offset
end
function _unflatten(nt::NamedTuple{K}, vec::AbstractVector, offset::Int) where {K}
    first_val, offset = _unflatten(first(values(nt)), vec, offset)
    rest_nt_template = NamedTuple{Base.tail(K)}(Base.tail(values(nt)))
    rest_nt, offset = _unflatten(rest_nt_template, vec, offset)
    return merge(NamedTuple{(first(K),)}((first_val,)), rest_nt), offset
end

function unflatten_from_vec(values::NamedTuple, vec::AbstractVector)
    expected_dim = _scalar_count(values)
    length(vec) == expected_dim || throw(
        DimensionMismatch(
            "vector length $(length(vec)) does not match expected dimension $expected_dim",
        ),
    )
    result, offset = _unflatten(values, vec, 1)
    offset == length(vec) + 1 || throw(
        DimensionMismatch(
            "vector length $(length(vec)) does not match expected dimension $expected_dim",
        ),
    )
    return result
end
