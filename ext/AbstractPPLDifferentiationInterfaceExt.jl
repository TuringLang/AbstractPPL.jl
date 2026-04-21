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

(p::DIPrepared)(x) = p.evaluator(x)

# Catch-all on `AbstractADType`: when DI is loaded but no native AbstractPPL
# extension matches the requested backend, route through DI. Native extensions
# (ForwardDiff, Enzyme, Mooncake, FiniteDifferences) define more specific
# methods that take precedence. NamedTuple inputs are intentionally not handled
# here; native extensions cover that path.
function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType,
    problem,
    x::AbstractVector{<:AbstractFloat};
    check_dims::Bool=true,
)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(
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
