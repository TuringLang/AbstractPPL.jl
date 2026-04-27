module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_jacobian_output, _assert_supported_output, _is_scalar_output
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

# The DI fallback calls this wrapper so only `x` is differentiated; in DynamicPPL the model and other evaluator state stay constant.
@inline _call_evaluator(x, evaluator) = evaluator(x)

struct DIPrepared{E,B,GP,JP} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    backend::B
    gradient_prep::GP
    jacobian_prep::JP
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
        gradient_prep = DI.prepare_gradient(
            _call_evaluator, adtype, x, DI.Constant(evaluator)
        )
        return DIPrepared(evaluator, adtype, gradient_prep, nothing)
    else
        _assert_jacobian_output(y)
        jacobian_prep = DI.prepare_jacobian(
            _call_evaluator, adtype, x, DI.Constant(evaluator)
        )
        return DIPrepared(evaluator, adtype, nothing, jacobian_prep)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared, x::AbstractVector{T}
) where {T<:Real}
    p.gradient_prep === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    val, grad = DI.value_and_gradient(
        _call_evaluator, p.gradient_prep, p.backend, x, DI.Constant(p.evaluator)
    )
    return (val, grad isa Vector{T} ? grad : Vector{T}(grad))
end

@inline function AbstractPPL.value_and_jacobian(p::DIPrepared, x::AbstractVector{<:Real})
    p.jacobian_prep === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    return DI.value_and_jacobian(
        _call_evaluator, p.jacobian_prep, p.backend, x, DI.Constant(p.evaluator)
    )
end

end # module
