module ADProblems

using ADTypes: AbstractADType

"""
    AbstractPrepared

Internal abstract supertype for all AD-prepared evaluators produced by AbstractPPL's
extension backends. Subtyping `AbstractPrepared` asserts that
`value_and_gradient(p, x)` is implemented; the LogDensityProblems extension
relies on this contract to advertise `LogDensityOrder{1}`.

Concrete subtypes must have an `evaluator` field whose value is callable on the
prepared input. In exchange they inherit the `(p::AbstractPrepared)(x)` forwarder.
"""
abstract type AbstractPrepared end

"""
    prepare(problem, values::NamedTuple)
    prepare(problem, x::AbstractVector{<:Real})
    prepare(adtype, problem, x::AbstractVector{<:Real}; check_dims::Bool=true)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector when it works with vector inputs. The
three-argument form, contributed by AD-backend extensions, additionally
prepares gradient or jacobian machinery for vector inputs.

The keyword argument `check_dims` (default `true`) controls whether the prepared
evaluator validates that inputs match the prototype used during preparation.
Pass `check_dims=false` when the caller guarantees input structure.
"""
function prepare end

# Downstream packages (e.g. DynamicPPL) pass already-callable objects,
# so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple) = problem
prepare(problem, x::AbstractVector{<:Real}) = problem

"""
    value_and_gradient(prepared, x::AbstractVector{<:Real})

Return `(value, gradient::AbstractVector)` for a scalar-valued evaluator prepared with a vector.
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
    VectorEvaluator{CheckInput}(f, dim)
    VectorEvaluator(f, dim)  # equivalent to `VectorEvaluator{true}(f, dim)`

Internal evaluator shape for scalar functions of a vector input.
Used by AbstractPPL's AD extensions; this is not part of the public API.

`CheckInput` controls whether the call method validates the input length. The default
(`true`) is the safe shape exposed to users via `prepared(x)`. AD extensions may
construct `VectorEvaluator{false}` for the inner callable handed to AD libraries,
where the input length is already guaranteed and the runtime check would otherwise
remain in the dual/shadow hot path.

A bare `VectorEvaluator` is *not* differentiable; gradient capability is the
contract of the wrapping `AbstractPrepared` returned by `prepare(adtype, ...)`.
"""
struct VectorEvaluator{CheckInput,F}
    f::F
    dim::Int
    function VectorEvaluator{CheckInput}(f::F, dim::Int) where {CheckInput,F}
        CheckInput isa Bool || throw(ArgumentError("`CheckInput` must be a Bool."))
        dim >= 0 || throw(ArgumentError("`dim` must be non-negative, got $dim."))
        return new{CheckInput,F}(f, dim)
    end
end

VectorEvaluator(f, dim::Int) = VectorEvaluator{true}(f, dim)

"""
    NamedTupleEvaluator{CheckInput}(f, inputspec)
    NamedTupleEvaluator(f, inputspec)  # equivalent to `NamedTupleEvaluator{true}(f, inputspec)`

Internal evaluator shape for scalar functions of a `NamedTuple` input with a
stable prototype. Used by AbstractPPL's AD extensions; this is not part of the
public API.

`CheckInput` controls whether the call method validates that an input `NamedTuple`
has the same type as the prototype captured during preparation.
"""
struct NamedTupleEvaluator{CheckInput,F,P<:NamedTuple}
    f::F
    inputspec::P
    function NamedTupleEvaluator{CheckInput}(
        f::F, inputspec::P
    ) where {CheckInput,F,P<:NamedTuple}
        CheckInput isa Bool || throw(ArgumentError("`CheckInput` must be a Bool."))
        return new{CheckInput,F,P}(f, inputspec)
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

# Reject integer vectors with a clear error rather than letting them flow into
# AD backends (which usually fail confusingly). Split per `CheckInput` to avoid
# an ambiguity with the `(::VectorEvaluator{true})(::AbstractVector)` method above.
function _reject_integer_input(::VectorEvaluator, x)
    throw(
        ArgumentError(
            "VectorEvaluator requires a vector of floating-point values, but received an `$(typeof(x))`. Convert to a floating-point vector (e.g. `Float64.(x)`) before calling.",
        ),
    )
end
(e::VectorEvaluator{true})(x::AbstractVector{<:Integer}) = _reject_integer_input(e, x)
(e::VectorEvaluator{false})(x::AbstractVector{<:Integer}) = _reject_integer_input(e, x)

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation. No-op when `e` was constructed with `CheckInput=false`.
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

(p::AbstractPrepared)(x) = p.evaluator(x)

_is_scalar_output(y) = y isa Number
_is_vector_output(y) = y isa AbstractVector

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

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, args, kwargs
        # `args` are argument types, not values (see `Base.Experimental.show_error_hints`).
        if exc.f === prepare && length(args) >= 1 && args[1] <: AbstractADType
            print(
                io,
                "\nCalling `prepare` with an AD backend requires loading the corresponding extension (e.g., `using DifferentiationInterface`).",
            )
        end
    end
end

end # module
