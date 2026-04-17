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

prepare(problem, values::NamedTuple) = throw(MethodError(prepare, (problem, values)))
function prepare(problem, x::AbstractVector{<:AbstractFloat})
    throw(MethodError(prepare, (problem, x)))
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
to return a pair `(problem, prototype)` suitable for `prepare(AutoFiniteDifferences(...), ...)`.
Additional keyword arguments are forwarded to `ADTypes.AutoFiniteDifferences`.
"""
function test_autograd end

function prepare_for_test_autograd end

function prepare_for_test_autograd(prepared, x)
    throw(
        ArgumentError(
            "`test_autograd` needs a finite-difference preparation path for $(typeof(prepared)). Define `prepare_for_test_autograd(prepared, x)` to return `(problem, prototype)`.",
        ),
    )
end

function test_autograd(
    prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, finite_difference_kwargs...
)
    val_ad, grad_ad = value_and_gradient(prepared, x)
    problem, prototype = prepare_for_test_autograd(prepared, x)
    fd_prepared = prepare(
        ADTypes.AutoFiniteDifferences(; finite_difference_kwargs...), problem, prototype
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

"""
    dimension(prepared)::Int

Return the number of scalar entries in the vector input expected by a prepared evaluator.
"""
function dimension end

end # module
