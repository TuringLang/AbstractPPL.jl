module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{E,F,B,C,P}
    evaluator::E
    f_vec::F
    backend::B
    prep::C
    values::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:DIPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::DIPrepared) = p.dim

function (p::DIPrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::DIPrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim ||
        throw(DimensionMismatch("expected vector of length $(p.dim), got $(length(x))"))
    return p.f_vec(x)
end

function AbstractPPL.prepare(adtype::ADTypes.AbstractADType, problem, values::NamedTuple)
    adtype isa Union{
        ADTypes.AutoFiniteDifferences,
        ADTypes.AutoForwardDiff,
        ADTypes.AutoEnzyme,
        ADTypes.AutoMooncake,
    } && throw(MethodError(AbstractPPL.prepare, (adtype, problem, values)))
    evaluator = AbstractPPL.prepare(problem, values)
    x0 = AbstractPPL.flatten_to_vec(values)
    f_vec = let evaluator = evaluator, values = values
        x -> evaluator(AbstractPPL.unflatten_from_vec(values, x))
    end
    prep = DI.prepare_gradient(f_vec, adtype, x0)
    return DIPrepared(evaluator, f_vec, adtype, prep, values, length(x0))
end

@inline function AbstractPPL.value_and_gradient(p::DIPrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(p.values, values)
    val, dx = DI.value_and_gradient(p.f_vec, p.prep, p.backend, x)
    grad_nt = AbstractPPL.unflatten_from_vec(p.values, values, dx)
    return (val, grad_nt)
end

end # module
