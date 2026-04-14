module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AbstractADType
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{E,F,B,C,P}
    evaluator::E
    f_vec::F
    backend::B
    prep::C
    prototype::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:DIPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::DIPrepared) = p.dim

function (p::DIPrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::DIPrepared)(x::AbstractVector)
    return p.f_vec(x)
end

function AbstractPPL.prepare(
    adtype::AbstractADType, problem, prototype::NamedTuple
)
    evaluator = AbstractPPL.prepare(problem, prototype)
    x0 = AbstractPPL.flatten_to_vec(prototype)
    f_vec = let evaluator = evaluator, prototype = prototype
        x -> evaluator(AbstractPPL.unflatten_from_vec(prototype, x))
    end
    prep = DI.prepare_gradient(f_vec, adtype, x0)
    return DIPrepared(evaluator, f_vec, adtype, prep, prototype, length(x0))
end

function AbstractPPL.value_and_gradient(p::DIPrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(values)
    val, dx = DI.value_and_gradient(p.f_vec, p.prep, p.backend, x)
    grad_nt = AbstractPPL.unflatten_from_vec(p.prototype, dx)
    return (val, grad_nt)
end

end # module
