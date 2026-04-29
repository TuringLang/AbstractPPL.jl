module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_jacobian_output, _assert_supported_output, _is_scalar_output
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

# Differentiate only `x`; the evaluator is passed as a `DI.Constant` context so
# that in DynamicPPL the model and other evaluator state stay constant.
@inline _call_evaluator(x, evaluator) = evaluator(x)

struct DIPrepared{E,B,F,GP,JP} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    backend::B
    target::F
    gradient_prep::GP
    jacobian_prep::JP
    use_context::Bool
end

function _prepare_gradient(adtype::ADTypes.AutoReverseDiff{true}, x, evaluator)
    # Compiled ReverseDiff only reuses a compiled tape on the one-argument path.
    target = Base.Fix2(_call_evaluator, evaluator)
    gradient_prep = DI.prepare_gradient(target, adtype, x)
    return target, gradient_prep, false
end

function _prepare_gradient(adtype::ADTypes.AbstractADType, x, evaluator)
    target = _call_evaluator
    gradient_prep = DI.prepare_gradient(target, adtype, x, DI.Constant(evaluator))
    return target, gradient_prep, true
end

function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType,
    problem,
    x::AbstractVector{<:Real};
    check_dims::Bool=true,
)
    raw = AbstractPPL.prepare(problem, x)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    # Empty inputs bypass `DI.prepare_*` (many backends fail on length-zero arrays);
    # `value_and_gradient`/`value_and_jacobian` short-circuit when `length(x) == 0`.
    length(x) == 0 &&
        return DIPrepared(evaluator, adtype, _call_evaluator, nothing, nothing, true)
    y = evaluator(x)
    _assert_supported_output(y)
    if _is_scalar_output(y)
        target, gradient_prep, use_context = _prepare_gradient(adtype, x, evaluator)
        return DIPrepared(evaluator, adtype, target, gradient_prep, nothing, use_context)
    else
        _assert_jacobian_output(y)
        jacobian_prep = DI.prepare_jacobian(
            _call_evaluator, adtype, x, DI.Constant(evaluator)
        )
        return DIPrepared(evaluator, adtype, _call_evaluator, nothing, jacobian_prep, true)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared, x::AbstractVector{T}
) where {T<:Real}
    length(x) == 0 && return (p.evaluator(x), T[])
    p.gradient_prep === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    val, grad = _value_and_gradient(p, x)
    # Some DI backends may return a non-`Vector` gradient; normalise.
    return (val, grad isa Vector ? grad : collect(grad))
end

@inline function _value_and_gradient(p::DIPrepared, x)
    return if p.use_context
        DI.value_and_gradient(
            p.target, p.gradient_prep, p.backend, x, DI.Constant(p.evaluator)
        )
    else
        DI.value_and_gradient(p.target, p.gradient_prep, p.backend, x)
    end
end

@inline function AbstractPPL.value_and_jacobian(p::DIPrepared, x::AbstractVector{<:Real})
    if length(x) == 0
        val = p.evaluator(x)
        return (val, similar(x, length(val), 0))
    end
    p.jacobian_prep === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    return DI.value_and_jacobian(
        p.target, p.jacobian_prep, p.backend, x, DI.Constant(p.evaluator)
    )
end

end # module
