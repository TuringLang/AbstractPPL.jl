"""
    DerivativeOrder{K}

Trait indicating the maximum derivative order supported by a prepared evaluator.
`K` must be 0, 1, or 2.
"""
struct DerivativeOrder{K}
    function DerivativeOrder{K}() where {K}
        K isa Int && 0 <= K <= 2 || throw(ArgumentError(
            "DerivativeOrder parameter must be 0, 1, or 2, got $K"
        ))
        return new{K}()
    end
end

"""
    capabilities(T::Type)
    capabilities(x)

Return the [`DerivativeOrder`](@ref) supported by a prepared evaluator type or instance.
Defaults to `DerivativeOrder{0}()`.
"""
capabilities(::Type) = DerivativeOrder{0}()
capabilities(x) = capabilities(typeof(x))

"""
    prepare(problem, prototype::NamedTuple)
    prepare(adtype::ADTypes.AbstractADType, problem, prototype::NamedTuple)

Prepare a callable evaluator for `problem`. The two-argument form performs structural
preparation only. The three-argument form additionally sets up AD for
[`value_and_gradient`](@ref).
"""
function prepare end

"""
    value_and_gradient(prepared, values::NamedTuple)

Return `(value, gradient)` where `gradient` has the same named structure as `values`.
Requires `capabilities(prepared) >= DerivativeOrder{1}()`.
"""
function value_and_gradient end

"""
    dimension(prepared)::Int

Return the number of scalar dimensions in the vector view of a prepared evaluator.
"""
function dimension end

## NamedTuple ↔ flat-vector utilities (used by AD extensions)

_scalar_count(::Real) = 1
_scalar_count(x::AbstractArray{<:Real}) = length(x)
function _scalar_count(nt::NamedTuple)
    n = 0
    for v in values(nt)
        n += _scalar_count(v)
    end
    return n
end

function _flatten!(vec::AbstractVector, s::Real, offset::Int)
    vec[offset] = s
    return offset + 1
end
function _flatten!(vec::AbstractVector, a::AbstractArray{<:Real}, offset::Int)
    n = length(a)
    for (j, v) in enumerate(a)
        vec[offset + j - 1] = v
    end
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
    vec = Vector{Float64}(undef, n)
    _flatten!(vec, nt, 1)
    return vec
end

function _unflatten(::Real, vec::AbstractVector, offset::Int)
    return vec[offset], offset + 1
end
function _unflatten(proto::AbstractArray{<:Real}, vec::AbstractVector, offset::Int)
    n = length(proto)
    result = reshape(vec[offset:offset + n - 1], size(proto))
    return result, offset + n
end
function _unflatten(proto::NamedTuple, vec::AbstractVector, offset::Int)
    rebuilt = Any[]
    for v in values(proto)
        val, offset = _unflatten(v, vec, offset)
        push!(rebuilt, val)
    end
    return NamedTuple{keys(proto)}(Tuple(rebuilt)), offset
end

function unflatten_from_vec(prototype::NamedTuple, vec::AbstractVector)
    result, _ = _unflatten(prototype, vec, 1)
    return result
end
