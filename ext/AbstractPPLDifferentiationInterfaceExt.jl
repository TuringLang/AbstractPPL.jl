module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_jacobian_output, _assert_supported_output, _is_scalar_output
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{E,B,GP,JP} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    backend::B
    gradient_prep::GP
    jacobian_prep::JP
end

# Catch-all for backends without a native AbstractPPL extension; native
# extensions take precedence via more-specific positional types.
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
        gradient_prep = DI.prepare_gradient(evaluator, adtype, x)
        return DIPrepared(evaluator, adtype, gradient_prep, nothing)
    else
        _assert_jacobian_output(y)
        jacobian_prep = DI.prepare_jacobian(evaluator, adtype, x)
        return DIPrepared(evaluator, adtype, nothing, jacobian_prep)
    end
end

@inline function AbstractPPL.value_and_gradient(p::DIPrepared, x::AbstractVector{<:Real})
    p.gradient_prep === nothing &&
        throw(ArgumentError("`value_and_gradient` requires a scalar-valued function."))
    return DI.value_and_gradient(p.evaluator, p.gradient_prep, p.backend, x)
end

@inline function AbstractPPL.value_and_jacobian(p::DIPrepared, x::AbstractVector{<:Real})
    p.jacobian_prep === nothing &&
        throw(ArgumentError("`value_and_jacobian` requires a vector-valued function."))
    return DI.value_and_jacobian(p.evaluator, p.jacobian_prep, p.backend, x)
end

AbstractPPL.ADProblems._supports_gradient(::DIPrepared{<:Any,<:Any,<:Any,Nothing}) = true

end # module
