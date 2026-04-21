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
    prepare(adtype, problem, values_or_vector; check_dims::Bool=true)

Prepare a callable evaluator for `problem`.

Use the two-argument form with a `NamedTuple` when the evaluator works with
named inputs, or with a vector of floating-point numbers when it works with
vector inputs. Automatic-differentiation backends extend this interface with
backend-specific three-argument methods.

The keyword argument `check_dims` (default `true`) controls whether the prepared
evaluator validates that inputs match the prototype used during preparation.
Pass `check_dims=false` when the caller guarantees input structure.
"""
function prepare end

# Identity defaults: AD backend extensions call the 2-arg form to obtain a
# callable from the problem. Downstream packages (e.g. DynamicPPL) pass
# already-callable objects, so the safe default is to return them unchanged.
prepare(problem, values::NamedTuple) = problem
prepare(problem, x::AbstractVector{<:AbstractFloat}) = problem

# Generic fallback: give a helpful error when the required AD package isn't loaded.
# AD backend extensions add more specific methods without overwriting this fallback.
#
# Note for downstream backends: every backend's `prepare(adtype, problem, x)` method
# must accept `; check_dims::Bool=true` itself. Julia's kwarg dispatch follows the
# most specific positional method, so a generic kwarg-accepting forwarder here
# cannot fall through to a backend's no-kwarg method. Each backend owns the kwarg
# on its own signature.
function prepare(adtype, problem, x::AbstractVector{<:AbstractFloat}; check_dims::Bool=true)
    throw(
        ArgumentError(
            "`prepare($(nameof(typeof(adtype)))(), ...)` requires loading the corresponding AD backend.",
        ),
    )
end
function prepare(adtype, problem, values::NamedTuple; check_dims::Bool=true)
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
    test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)

Compare `value_and_gradient(prepared, x)` against a finite-difference reference.
Throws an informative error on mismatch. Returns `nothing`.

Backends should define `AbstractPPL.ADProblems.prepare_for_test_autograd(prepared, x)`
returning `(problem, prototype, fdm)` to enable this helper.
"""
function test_autograd end

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
"""
struct VectorEvaluator{Checked,F}
    f::F
    dim::Int
    function VectorEvaluator{Checked}(f::F, dim::Int) where {Checked,F}
        Checked isa Bool || throw(ArgumentError("`Checked` must be a Bool."))
        return new{Checked,F}(f, dim)
    end
end

VectorEvaluator(f, dim::Int) = VectorEvaluator{true}(f, dim)

"""
    NamedTupleEvaluator{Checked}(f, prototype)
    NamedTupleEvaluator(f, prototype)  # equivalent to `NamedTupleEvaluator{true}(f, prototype)`

Internal evaluator shape for scalar functions of a `NamedTuple` input with a
stable prototype. Used by AbstractPPL's AD extensions; this is not part of the
public API.

`Checked` controls whether wrapper code (via [`_assert_namedtuple_shape`](@ref))
validates that an input `NamedTuple` has the same type as the prototype captured
during preparation.
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

function prepare_for_test_autograd(prepared, x)
    throw(
        ArgumentError(
            "`test_autograd` needs a finite-difference preparation path for $(typeof(prepared)). Define `prepare_for_test_autograd(prepared, x)` to return `(problem, prototype, fdm)`.",
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

(e::NamedTupleEvaluator)(values::NamedTuple) = e.f(values)

(e::VectorEvaluator{true})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))
(e::VectorEvaluator{false})(x::AbstractVector{<:Integer}) = throw(MethodError(e, (x,)))

"""
    _assert_namedtuple_shape(e::NamedTupleEvaluator, values)

Throw `ArgumentError` unless `values` has the same type as the prototype captured
during preparation. No-op when `e` was constructed with `Checked=false`.
Internal helper shared by AD extensions.
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

function test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)
    val_ad, grad_ad = value_and_gradient(prepared, x)
    problem, prototype, fdm = prepare_for_test_autograd(prepared, x)
    fd_prepared = prepare(ADTypes.AutoFiniteDifferences(; fdm), problem, prototype)
    val_fd, grad_fd = value_and_gradient(fd_prepared, x)

    isapprox(val_ad, val_fd; atol=atol, rtol=rtol) || throw(
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
