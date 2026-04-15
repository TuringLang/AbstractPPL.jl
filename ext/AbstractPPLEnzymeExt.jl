module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,F,P}
    evaluator::E
    f_vec::F
    values::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = p.dim

function (p::EnzymePrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::EnzymePrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim ||
        throw(DimensionMismatch("expected vector of length $(p.dim), got $(length(x))"))
    return p.f_vec(x)
end

function AbstractPPL.prepare(::AutoEnzyme, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    x0 = AbstractPPL.flatten_to_vec(values)
    f_vec = let evaluator = evaluator, values = values
        x -> evaluator(AbstractPPL.unflatten_from_vec(values, x))
    end
    return EnzymePrepared(evaluator, f_vec, values, length(x0))
end

@inline function AbstractPPL.value_and_gradient(p::EnzymePrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(p.values, values)
    dx = zero(x)
    result = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(p.f_vec),
        Enzyme.Active,
        Enzyme.Duplicated(x, dx),
    )
    val = result[2]  # autodiff(ReverseWithPrimal, ...) returns ((adjoints...,), primal)
    grad_nt = AbstractPPL.unflatten_from_vec(p.values, values, dx)
    return (val, grad_nt)
end

end # module
