module ADProblems

@static if VERSION >= v"1.11.0"
    eval(Meta.parse("public prepare, value_and_gradient, value_and_jacobian"))
end

"""
    AbstractPrepared

Internal abstract supertype for all AD-prepared evaluators produced by AbstractPPL's
extension backends.

Concrete subtypes must have an `evaluator` field (`VectorEvaluator` or
`NamedTupleEvaluator`). In exchange they inherit the callable forwarder automatically.
"""
abstract type AbstractPrepared end

"""
    prepare(problem, values::NamedTuple)
    prepare(problem, x::AbstractVector{<:Real})
    prepare(adtype, problem, values_or_vector; check_dims::Bool=true)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector when it works with vector inputs. Automatic
differentiation backends extend this interface with
backend-specific three-argument methods.

The keyword argument `check_dims` (default `true`) controls whether the prepared
evaluator validates that inputs match the prototype used during preparation.
Pass `check_dims=false` when the caller guarantees input structure.
"""
function prepare end

_is_scalar_output(y) = y isa Number
_is_vector_output(y) = y isa AbstractVector

function _assert_gradient_output(y)
    _is_scalar_output(y) || throw(
        ArgumentError(
            "`value_and_gradient` requires a scalar-valued function; got $(typeof(y))."
        ),
    )
    return nothing
end

function _assert_jacobian_output(y)
    _is_vector_output(y) || throw(
        ArgumentError(
            "`value_and_jacobian` requires the prepared function to return an AbstractVector; got $(typeof(y)).",
        ),
    )
    return nothing
end

function _assert_supported_output(y)
    (_is_scalar_output(y) || _is_vector_output(y)) || throw(
        ArgumentError(
            "A prepared AD evaluator must return a scalar or AbstractVector; got $(typeof(y)).",
        ),
    )
    return nothing
end

# Downstream packages (e.g. DynamicPPL) pass already-callable objects,
# so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple) = problem
prepare(problem, x::AbstractVector{<:Real}) = problem

"""
    value_and_gradient(prepared, x::AbstractVector{<:Real})

Return `(value, gradient::AbstractVector)` for a scalar-valued evaluator prepared with a vector.

A NamedTuple overload is also available when the evaluator was prepared with a
`NamedTuple` prototype.
"""
function value_and_gradient end

"""
    value_and_jacobian(prepared, x::AbstractVector{<:Real})

Return `(value::AbstractVector, jacobian::AbstractMatrix)` for a vector-valued
evaluator prepared with a vector. The returned `jacobian` has shape
`(length(value), length(x))`.
"""
function value_and_jacobian end

"""
    VectorEvaluator{Validate}(f, dim)
    VectorEvaluator(f, dim)  # equivalent to `VectorEvaluator{true}(f, dim)`

Internal evaluator shape for scalar functions of a vector input.
Used by AbstractPPL's AD extensions; this is not part of the public API.

`Validate` controls whether the call method validates the input length. The default
(`true`) is the safe shape exposed to users via `prepared(x)`. AD extensions may
construct `VectorEvaluator{false}` for the inner callable handed to AD libraries,
where the input length is already guaranteed and the runtime check would otherwise
remain in the dual/shadow hot path.

The `Trivial` type parameter is `true` iff `dim == 0`. Because `Trivial` is
determined at construction time from a runtime `dim`, a `VectorEvaluator` built
with a runtime-unknown `dim` will have `Trivial` inferred as `Bool` rather than a
concrete `true` or `false`, which affects type stability of dispatch on the
`{Validate,true}` overloads.
"""
struct VectorEvaluator{Validate,Trivial,F}
    f::F
    dim::Int
    function VectorEvaluator{Validate}(f::F, dim::Int) where {Validate,F}
        Validate isa Bool || throw(ArgumentError("`Validate` must be a Bool."))
        dim >= 0 || throw(ArgumentError("`dim` must be non-negative, got $dim."))
        return new{Validate,dim == 0,F}(f, dim)
    end
end

VectorEvaluator(f, dim::Int) = VectorEvaluator{true}(f, dim)

# Trivial (dim == 0) evaluators are a complete prepared evaluator on their own:
# both gradient and jacobian are well-defined, and all backends can route
# zero-dimensional inputs through this shape.
function value_and_gradient(e::VectorEvaluator{V,true}, x::AbstractVector{<:Real}) where {V}
    length(x) == 0 ||
        throw(DimensionMismatch("Expected an empty vector, but got length $(length(x))."))
    val = e.f(x)
    _assert_gradient_output(val)
    return (val, similar(x))
end

function value_and_jacobian(e::VectorEvaluator{V,true}, x::AbstractVector{<:Real}) where {V}
    length(x) == 0 ||
        throw(DimensionMismatch("Expected an empty vector, but got length $(length(x))."))
    val = e.f(x)
    _assert_jacobian_output(val)
    return (val, similar(x, length(val), 0))
end

"""
    NamedTupleEvaluator{Validate}(f, inputspec)
    NamedTupleEvaluator(f, inputspec)  # equivalent to `NamedTupleEvaluator{true}(f, inputspec)`

Internal evaluator shape for scalar functions of a `NamedTuple` input with a
stable prototype. Used by AbstractPPL's AD extensions; this is not part of the
public API.

`Validate` controls whether the call method validates that an input `NamedTuple`
has the same type as the prototype captured during preparation.
"""
struct NamedTupleEvaluator{Validate,F,P<:NamedTuple}
    f::F
    inputspec::P
    function NamedTupleEvaluator{Validate}(
        f::F, inputspec::P
    ) where {Validate,F,P<:NamedTuple}
        Validate isa Bool || throw(ArgumentError("`Validate` must be a Bool."))
        return new{Validate,F,P}(f, inputspec)
    end
end

NamedTupleEvaluator(f, inputspec::NamedTuple) = NamedTupleEvaluator{true}(f, inputspec)

function (e::VectorEvaluator{true})(x::AbstractVector)
    length(x) == e.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(e.dim), but got length $(length(x))."
        ),
    )
    return e.f(x)
end

(e::VectorEvaluator{false})(x::AbstractVector) = e.f(x)

function (e::NamedTupleEvaluator{true})(values::NamedTuple)
    _assert_namedtuple_shape(e, values)
    return e.f(values)
end
(e::NamedTupleEvaluator{false})(values::NamedTuple) = e.f(values)

# Two separate overloads rather than one `(::VectorEvaluator)(::AbstractVector{<:Integer})`
# to avoid ambiguity with `(::VectorEvaluator{true})(::AbstractVector)`.
(e::VectorEvaluator{true})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))
(e::VectorEvaluator{false})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation. No-op when `e` was constructed with `Validate=false`.
"""
function _assert_namedtuple_shape(e::NamedTupleEvaluator{true}, values)
    typeof(values) === typeof(e.inputspec) || throw(
        ArgumentError(
            "Expected the same NamedTuple structure that was used to prepare this evaluator.",
        ),
    )
    return nothing
end
_assert_namedtuple_shape(::NamedTupleEvaluator{false}, _) = nothing

# These rely on the `evaluator` field contract, so they are defined after the evaluator types.
(p::AbstractPrepared)(x) = p.evaluator(x)

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, args, kwargs
        if exc.f === prepare && length(args) >= 3
            print(
                io,
                "\nCalling `prepare` with an AD backend requires loading the corresponding extension (e.g., `using DifferentiationInterface`).",
            )
        end
    end
end

end # module
