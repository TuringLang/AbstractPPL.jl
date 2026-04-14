module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoMooncake
using Mooncake: Mooncake

struct MooncakePrepared{E,F,C,P}
    evaluator::E
    f_vec::F
    cache::C
    prototype::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:MooncakePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::MooncakePrepared) = p.dim

function (p::MooncakePrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::MooncakePrepared)(x::AbstractVector)
    length(x) == p.dim ||
        throw(DimensionMismatch("expected vector of length $(p.dim), got $(length(x))"))
    return p.f_vec(x)
end

function AbstractPPL.prepare(adtype::AutoMooncake, problem, prototype::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, prototype)
    x0 = AbstractPPL.flatten_to_vec(prototype)
    f_vec = let evaluator = evaluator, prototype = prototype
        x -> evaluator(AbstractPPL.unflatten_from_vec(prototype, x))
    end
    cache = Mooncake.prepare_gradient_cache(f_vec, x0; config=adtype.config)
    return MooncakePrepared(evaluator, f_vec, cache, prototype, length(x0))
end

@inline function AbstractPPL.value_and_gradient(p::MooncakePrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(values)
    val, (_, dx) = Mooncake.value_and_gradient!!(p.cache, p.f_vec, x)
    grad_nt = AbstractPPL.unflatten_from_vec(p.prototype, dx)
    return (val, grad_nt)
end

end # module
