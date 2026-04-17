module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{E,B,C}
    evaluator::E
    backend::B
    prep::C
end

AbstractPPL.capabilities(::Type{<:DIPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::DIPrepared) = AbstractPPL.dimension(p.evaluator)

function (p::DIPrepared)(x)
    return p.evaluator(x)
end

function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator(
        AbstractPPL.prepare(problem, x), length(x)
    )
    prep = DI.prepare_gradient(evaluator, adtype, x)
    return DIPrepared(evaluator, adtype, prep)
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared, x::AbstractVector{<:AbstractFloat}
)
    return DI.value_and_gradient(p.evaluator, p.prep, p.backend, x)
end

end # module
