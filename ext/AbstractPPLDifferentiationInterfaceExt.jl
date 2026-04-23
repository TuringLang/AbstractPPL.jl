module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems: _check_mode
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{Mode,E,B,C} <: AbstractPPL.ADProblems.AbstractPrepared{Mode}
    evaluator::E
    backend::B
    prep::C
    function DIPrepared{Mode}(evaluator::E, backend::B, prep::C) where {Mode,E,B,C}
        return new{Mode,E,B,C}(evaluator, backend, prep)
    end
end

# Catch-all for backends without a native AbstractPPL extension; native
# extensions take precedence via more-specific positional types.
# NamedTuple inputs are not handled here; native extensions cover that path.
function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
    mode::Symbol=:gradient,
)
    _check_mode(mode)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
        AbstractPPL.prepare(problem, x), length(x)
    )
    if mode === :gradient
        prep = DI.prepare_gradient(evaluator, adtype, x)
        return DIPrepared{:gradient}(evaluator, adtype, prep)
    else
        prep = DI.prepare_jacobian(evaluator, adtype, x)
        return DIPrepared{:jacobian}(evaluator, adtype, prep)
    end
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared{:gradient}, x::AbstractVector{<:AbstractFloat}
)
    return DI.value_and_gradient(p.evaluator, p.prep, p.backend, x)
end

@inline function AbstractPPL.value_and_jacobian(
    p::DIPrepared{:jacobian}, x::AbstractVector{<:AbstractFloat}
)
    return DI.value_and_jacobian(p.evaluator, p.prep, p.backend, x)
end

end # module
