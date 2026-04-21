module ADProblems

using ADTypes: ADTypes

export DerivativeOrder, capabilities, prepare, value_and_gradient, test_autograd, dimension

"""
    DerivativeOrder{K}

Represents the highest derivative order supported by a prepared evaluator.
`K` must be 0, 1, or 2.
"""
struct DerivativeOrder{K}
    function DerivativeOrder{K}() where {K}
        K isa Int && 0 <= K <= 2 ||
            throw(ArgumentError("DerivativeOrder must be 0, 1, or 2, but got $K."))
        return new{K}()
    end
end

Base.isless(::DerivativeOrder{K}, ::DerivativeOrder{L}) where {K,L} = K < L

"""
    capabilities(T::Type)
    capabilities(x)

Return the [`DerivativeOrder`](@ref) supported by a prepared evaluator type or instance.
Prepared evaluators default to [`DerivativeOrder{0}`](@ref) unless they define higher-order support explicitly.
"""
capabilities(::Type) = DerivativeOrder{0}()
capabilities(x) = capabilities(typeof(x))

"""
    prepare(problem, values::NamedTuple)
    prepare(problem, x::AbstractVector{<:AbstractFloat})

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector of floating-point numbers when it works with
vector inputs. Automatic-differentiation backends extend this interface with
backend-specific three-argument methods.
"""
function prepare end

# Identity defaults: AD backend extensions call the 2-arg form to obtain a
# callable from the problem. Downstream packages (e.g. DynamicPPL) pass
# already-callable objects, so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple) = problem
prepare(problem, x::AbstractVector{<:AbstractFloat}) = problem

# Generic fallback: give a helpful error when the required AD package isn't loaded.
# AD backend extensions add more specific methods without overwriting this fallback.
function prepare(adtype, problem, x::AbstractVector{<:AbstractFloat})
    throw(
        ArgumentError(
            "`prepare($(nameof(typeof(adtype)))(), ...)` requires loading the corresponding AD backend.",
        ),
    )
end
function prepare(adtype, problem, values::NamedTuple)
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

Requires `capabilities(prepared) >= DerivativeOrder{1}()`.
"""
function value_and_gradient end

function value_and_gradient(prepared, x::AbstractVector{<:AbstractFloat})
    throw(
        ArgumentError(
            "This evaluator does not support gradients for a vector of floating-point numbers.",
        ),
    )
end

"""
    test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, finite_difference_kwargs...)

Compare `value_and_gradient(prepared, x)` against a finite-difference reference
computed via `value_and_gradient(prepare(AutoFiniteDifferences(...), problem, x), x)`.
Throws an informative error on mismatch. Returns `nothing`.

Backends that want this helper should define `prepare_for_test_autograd(prepared, x)`
to return `(problem, prototype, fdm)` suitable for `prepare(AutoFiniteDifferences(...), ...)`.
Additional keyword arguments are forwarded to `ADTypes.AutoFiniteDifferences`.
"""
function test_autograd end

"""
    dimension(prepared)::Int

Return the number of scalar entries in the vector input expected by a prepared evaluator.
"""
function dimension end

"""
    VectorEvaluator(f, dim)

Internal evaluator shape for scalar functions of a floating-point vector input.
Used by AbstractPPL's AD extensions; this is not part of the public API.
"""
struct VectorEvaluator{F}
    f::F
    dim::Int
end

"""
    NamedTupleEvaluator(f, prototype)

Internal evaluator shape for scalar functions of a `NamedTuple` input with a
stable prototype. Used by AbstractPPL's AD extensions; this is not part of the
public API.
"""
struct NamedTupleEvaluator{F,P<:NamedTuple}
    f::F
    inputspec::P
end

dimension(e::VectorEvaluator) = e.dim
function dimension(::NamedTupleEvaluator)
    throw(
        ArgumentError(
            "`dimension` is only available for evaluators prepared with a vector of floating-point numbers.",
        ),
    )
end

function prepare_for_test_autograd end

function prepare_for_test_autograd(prepared, x)
    throw(
        ArgumentError(
            "`test_autograd` needs a finite-difference preparation path for $(typeof(prepared)). Define `prepare_for_test_autograd(prepared, x)` to return `(problem, prototype, fdm)`.",
        ),
    )
end

function (e::VectorEvaluator)(x::AbstractVector)
    length(x) == e.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(e.dim), but got length $(length(x))."
        ),
    )
    return e.f(x)
end

(e::NamedTupleEvaluator)(values::NamedTuple) = e.f(values)

function (e::NamedTupleEvaluator)(x::AbstractVector)
    throw(MethodError(e, (x,)))
end

function (e::VectorEvaluator)(x::AbstractVector{<:Integer})
    throw(MethodError(e, (x,)))
end

function (e::VectorEvaluator)(x)
    throw(MethodError(e, (x,)))
end

function (e::NamedTupleEvaluator)(x)
    throw(MethodError(e, (x,)))
end

function test_autograd(
    prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, finite_difference_kwargs...
)
    val_ad, grad_ad = value_and_gradient(prepared, x)
    problem, prototype, fdm = prepare_for_test_autograd(prepared, x)
    fd_prepared = prepare(
        ADTypes.AutoFiniteDifferences(; fdm, finite_difference_kwargs...),
        problem,
        prototype,
    )
    val_fd, grad_fd = value_and_gradient(fd_prepared, x)

    isapprox(val_ad, val_fd) || throw(
        ArgumentError(
            "Value mismatch against finite differences: got $val_ad, expected $val_fd."
        ),
    )
    isapprox(grad_ad, grad_fd; atol=atol, rtol=rtol) || throw(
        ArgumentError(
            "Gradient mismatch against finite differences with atol=$atol and rtol=$rtol.",
        ),
    )
    return nothing
end

end # module
