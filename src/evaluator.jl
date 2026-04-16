using ADTypes: ADTypes

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
    test_grad(f, x::AbstractVector{<:AbstractFloat})

Return a finite-difference reference gradient for a scalar-valued callable `f`
evaluated at the vector input `x`.

If the FiniteDifferences extension is not loaded, this warns and returns `nothing`.
"""
function test_grad end

function test_grad(f, x)
    @warn "Finite-difference reference gradients require `using FiniteDifferences`; skipping test_grad."
    return nothing
end

"""
    dimension(prepared)::Int

Return the number of scalar entries in the vector input expected by a prepared evaluator.
"""
function dimension end
