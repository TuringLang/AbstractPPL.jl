module ADProblems

using ADTypes: ADTypes

export prepare, value_and_gradient, value_and_jacobian, test_autograd

"""
    AbstractPrepared{Mode}

Internal abstract supertype for all AD-prepared evaluators produced by AbstractPPL's
extension backends. `Mode` is `:gradient` or `:jacobian`.

Concrete subtypes must have an `evaluator` field (`VectorEvaluator` or
`NamedTupleEvaluator`). In exchange they inherit `dimension` and
the callable forwarder automatically.
"""
abstract type AbstractPrepared{Mode} end

"""
    prepare(problem, values::NamedTuple)
    prepare(problem, x::AbstractVector{<:AbstractFloat})
    prepare(adtype, problem, values_or_vector; check_dims::Bool=true, mode::Symbol=:gradient)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector of floating-point numbers when it works with
vector inputs. Automatic-differentiation backends extend this interface with
backend-specific three-argument methods.

The keyword argument `check_dims` (default `true`) controls whether the prepared
evaluator validates that inputs match the prototype used during preparation.
Pass `check_dims=false` when the caller guarantees input structure.

The keyword argument `mode` (default `:gradient`) selects what derivative the
prepared evaluator can compute. Supported values are `:gradient` (scalar-valued
`f`, used with [`value_and_gradient`](@ref)) and `:jacobian` (vector-valued `f`,
used with [`value_and_jacobian`](@ref)). Jacobian preparation is only supported
on the vector-input path.
"""
function prepare end

function _check_mode(mode::Symbol)
    mode === :gradient ||
        mode === :jacobian ||
        throw(ArgumentError("`mode` must be `:gradient` or `:jacobian`, got `:$(mode)`."))
    return nothing
end

function _check_namedtuple_mode(mode::Symbol)
    mode === :gradient || throw(
        ArgumentError(
            "`mode=:$(mode)` is only supported on the vector-input path; NamedTuple inputs are gradient-only.",
        ),
    )
    return nothing
end

# Identity defaults: AD backend extensions call the 2-arg form to obtain a
# callable from the problem. Downstream packages (e.g. DynamicPPL) pass
# already-callable objects, so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple) = problem
prepare(problem, x::AbstractVector{<:AbstractFloat}) = problem

# Each backend owns `check_dims` and `mode` on its own signature: kwarg dispatch
# follows the most specific positional method, so this fallback cannot forward to
# a backend method that doesn't itself accept these kwargs.
function prepare(
    adtype,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_mode(mode)
    throw(
        ArgumentError(
            "`prepare($(nameof(typeof(adtype)))(), ...)` requires loading the corresponding AD backend.",
        ),
    )
end
function prepare(
    adtype, problem, values::NamedTuple; check_dims::Bool=true, mode::Symbol=:gradient
)
    _check_namedtuple_mode(mode)
    throw(
        ArgumentError(
            "`prepare($(nameof(typeof(adtype)))(), ...)` requires loading the corresponding AD backend.",
        ),
    )
end

"""
    value_and_gradient(prepared, x::AbstractVector{<:AbstractFloat})

Return `(value, gradient::AbstractVector)` for an evaluator prepared with a
vector of floating-point numbers.

Requires an evaluator prepared with `mode=:gradient`. A NamedTuple overload is
also available when the evaluator was prepared with a `NamedTuple` prototype.
"""
function value_and_gradient end

function value_and_gradient(prepared, x::AbstractVector{<:AbstractFloat})
    throw(
        ArgumentError(
            "This evaluator does not support gradients for a vector of floating-point numbers. Re-prepare it with `mode=:gradient`.",
        ),
    )
end

"""
    value_and_jacobian(prepared, x::AbstractVector{<:AbstractFloat})

Return `(value::AbstractVector, jacobian::AbstractMatrix)` for an evaluator
prepared with a vector of floating-point numbers and `mode=:jacobian`.
The returned `jacobian` has shape `(length(value), length(x))`.

Requires an evaluator prepared with `mode=:jacobian`.
"""
function value_and_jacobian end

function value_and_jacobian(prepared, x::AbstractVector{<:AbstractFloat})
    throw(
        ArgumentError(
            "This evaluator does not support jacobians. Re-prepare it with `mode=:jacobian`.",
        ),
    )
end

"""
    test_autograd(prepared, x; atol=1e-5, rtol=1e-5)

Compare `value_and_gradient(prepared, x)` against a finite-difference reference.
Throws an informative error on mismatch. Returns `nothing`.

Errors if `prepared` was built with `mode=:jacobian`; only gradient-mode
evaluators are supported.

Requires loading FiniteDifferences so the extension-backed implementation is available.
"""
function test_autograd(prepared, x; atol=1e-5, rtol=1e-5)
    _assert_gradient_capability(prepared)
    throw(
        ArgumentError(
            "`test_autograd` requires loading FiniteDifferences to activate the AbstractPPLFiniteDifferencesExt implementation.",
        ),
    )
end

function _assert_gradient_capability(prepared)
    prepared isa AbstractPrepared{:jacobian} && throw(
        ArgumentError(
            "`test_autograd` only supports gradient-mode evaluators; got an evaluator prepared with `mode=:jacobian`.",
        ),
    )
    return nothing
end

"""
    dimension(prepared)::Int

Return the number of scalar entries in the vector input expected by a prepared evaluator.
"""
function dimension end

"""
    VectorEvaluator{Checked}(f, dim)
    VectorEvaluator(f, dim)  # equivalent to `VectorEvaluator{true}(f, dim)`

Internal evaluator shape for scalar functions of a floating-point vector input.
Used by AbstractPPL's AD extensions; this is not part of the public API.

`Checked` controls whether the call method validates the input length. The default
(`true`) is the safe shape exposed to users via `prepared(x)`. AD extensions may
construct `VectorEvaluator{false}` for the inner callable handed to AD libraries,
where the input length is already guaranteed and the runtime check would otherwise
remain in the dual/shadow hot path.

The `Trivial` type parameter is `true` iff `dim == 0`.
"""
struct VectorEvaluator{Checked,Trivial,F}
    f::F
    dim::Int
    function VectorEvaluator{Checked}(f::F, dim::Int) where {Checked,F}
        Checked isa Bool || throw(ArgumentError("`Checked` must be a Bool."))
        dim >= 0 || throw(ArgumentError("`dim` must be non-negative, got $dim."))
        return new{Checked,dim == 0,F}(f, dim)
    end
end

VectorEvaluator(f, dim::Int) = VectorEvaluator{true}(f, dim)

# Trivial (dim == 0) evaluators are a complete prepared evaluator on their own:
# both gradient and jacobian are well-defined, and all backends can route
# zero-dimensional inputs through this shape.
function value_and_gradient(
    e::VectorEvaluator{C,true}, x::AbstractVector{<:AbstractFloat}
) where {C}
    length(x) == 0 ||
        throw(DimensionMismatch("Expected an empty vector, but got length $(length(x))."))
    return (e.f(x), similar(x))
end

function value_and_jacobian(
    e::VectorEvaluator{C,true}, x::AbstractVector{<:AbstractFloat}
) where {C}
    length(x) == 0 ||
        throw(DimensionMismatch("Expected an empty vector, but got length $(length(x))."))
    val = e.f(x)
    val isa AbstractVector || throw(
        ArgumentError(
            "`mode=:jacobian` requires `f(x)` to return an AbstractVector; got $(typeof(val)).",
        ),
    )
    return (val, similar(x, length(val), 0))
end

"""
    NamedTupleEvaluator{Checked}(f, prototype)
    NamedTupleEvaluator(f, prototype)  # equivalent to `NamedTupleEvaluator{true}(f, prototype)`

Internal evaluator shape for scalar functions of a `NamedTuple` input with a
stable prototype. Used by AbstractPPL's AD extensions; this is not part of the
public API.

`Checked` controls whether the call method validates that an input `NamedTuple`
has the same type as the prototype captured during preparation.
"""
struct NamedTupleEvaluator{Checked,F,P<:NamedTuple}
    f::F
    inputspec::P
    function NamedTupleEvaluator{Checked}(
        f::F, inputspec::P
    ) where {Checked,F,P<:NamedTuple}
        Checked isa Bool || throw(ArgumentError("`Checked` must be a Bool."))
        return new{Checked,F,P}(f, inputspec)
    end
end

NamedTupleEvaluator(f, inputspec::NamedTuple) = NamedTupleEvaluator{true}(f, inputspec)

dimension(e::VectorEvaluator) = e.dim
function dimension(::NamedTupleEvaluator)
    throw(
        ArgumentError(
            "`dimension` is only available for evaluators prepared with a vector of floating-point numbers.",
        ),
    )
end

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

(e::VectorEvaluator{true})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))
(e::VectorEvaluator{false})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation. No-op when `e` was constructed with `Checked=false`.
Internal helper; called at `value_and_gradient` entry points in AD extensions.
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

# Shared behaviours for all AbstractPrepared subtypes. These rely on the
# `evaluator` field contract, so they are defined after the evaluator types.
dimension(p::AbstractPrepared) = dimension(p.evaluator)
(p::AbstractPrepared)(x) = p.evaluator(x)

end # module
