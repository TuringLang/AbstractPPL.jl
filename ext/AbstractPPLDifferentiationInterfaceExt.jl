module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_jacobian_output, _assert_supported_output, _is_scalar_output
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

# The DI fallback calls this wrapper so only `x` is differentiated; in DynamicPPL the model and other evaluator state stay constant.
@inline _call_evaluator(x, evaluator) = evaluator(x)

struct DIPrepared{UseContext,E,B,F,GP,JP} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    backend::B
    target::F
    gradient_prep::GP
    jacobian_prep::JP
end

function DIPrepared(
    ::Val{UseContext}, evaluator, backend, target, gradient_prep, jacobian_prep
) where {UseContext}
    return DIPrepared{
        UseContext,
        typeof(evaluator),
        typeof(backend),
        typeof(target),
        typeof(gradient_prep),
        typeof(jacobian_prep),
    }(
        evaluator, backend, target, gradient_prep, jacobian_prep
    )
end

function _prepare_gradient(adtype::ADTypes.AutoReverseDiff{true}, x, evaluator)
    # Compiled ReverseDiff only reuses a compiled tape on the one-argument path.
    target = Base.Fix2(_call_evaluator, evaluator)
    gradient_prep = DI.prepare_gradient(target, adtype, x)
    return target, gradient_prep, Val(false)
end

function _prepare_gradient(adtype::ADTypes.AbstractADType, x, evaluator)
    target = _call_evaluator
    gradient_prep = DI.prepare_gradient(target, adtype, x, DI.Constant(evaluator))
    return target, gradient_prep, Val(true)
end

# Catch-all for backends without a native AbstractPPL extension;
# native extensions take precedence via more-specific positional types.
# NamedTuple inputs are not handled here; native extensions cover that path.
function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    y = evaluator(x)
    _assert_supported_output(y)
    if _is_scalar_output(y)
        target, gradient_prep, use_context = _prepare_gradient(adtype, x, evaluator)
        return DIPrepared(use_context, evaluator, adtype, target, gradient_prep, nothing)
    else
        _assert_jacobian_output(y)
        jacobian_prep = DI.prepare_jacobian(
            _call_evaluator, adtype, x, DI.Constant(evaluator)
        )
        return DIPrepared(
            Val(true), evaluator, adtype, _call_evaluator, nothing, jacobian_prep
        )
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared, x::AbstractVector{T}
) where {T<:Real}
    p.gradient_prep === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    val, grad = _value_and_gradient(p, x)
    return (val, grad isa Vector{T} ? grad : Vector{T}(grad))
end

@inline function _value_and_gradient(p::DIPrepared{true}, x)
    return DI.value_and_gradient(
        p.target, p.gradient_prep, p.backend, x, DI.Constant(p.evaluator)
    )
end

@inline function _value_and_gradient(p::DIPrepared{false}, x)
    return DI.value_and_gradient(p.target, p.gradient_prep, p.backend, x)
end

@inline function AbstractPPL.value_and_jacobian(p::DIPrepared, x::AbstractVector{<:Real})
    p.jacobian_prep === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    return _value_and_jacobian(p, x)
end

# Only `{true}` is reachable: the jacobian path in `prepare` always sets
# `use_context = Val(true)`, so a `{false}` overload would be dead code.
@inline function _value_and_jacobian(p::DIPrepared{true}, x)
    return DI.value_and_jacobian(
        p.target, p.jacobian_prep, p.backend, x, DI.Constant(p.evaluator)
    )
end

end # module
