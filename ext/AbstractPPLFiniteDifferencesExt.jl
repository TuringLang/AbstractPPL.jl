module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoFiniteDifferences
using FiniteDifferences: FiniteDifferences

struct FDPrepared{E,F,M,P}
    evaluator::E
    f_vec::F
    fdm::M
    values::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:FDPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::FDPrepared) = p.dim

function (p::FDPrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::FDPrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim ||
        throw(DimensionMismatch("expected vector of length $(p.dim), got $(length(x))"))
    return p.f_vec(x)
end

function AbstractPPL.prepare(adtype::AutoFiniteDifferences, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    x0 = AbstractPPL.flatten_to_vec(values)
    f_vec = let evaluator = evaluator, values = values
        x -> evaluator(AbstractPPL.unflatten_from_vec(values, x))
    end
    return FDPrepared(evaluator, f_vec, adtype.fdm, values, length(x0))
end

function AbstractPPL.value_and_gradient(p::FDPrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(p.values, values)
    val = p.f_vec(x)
    grad_vec = FiniteDifferences.grad(p.fdm, p.f_vec, x)[1]
    grad_nt = AbstractPPL.unflatten_from_vec(p.values, values, grad_vec)
    return (val, grad_nt)
end

end # module
