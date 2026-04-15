module AbstractPPLMooncakeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoMooncake
using Mooncake: Mooncake

struct MooncakePrepared{E,F,C,P}
    evaluator::E
    f_vec::F
    cache::C
    values::P
end

AbstractPPL.capabilities(::Type{<:MooncakePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::MooncakePrepared) = AbstractPPL._scalar_count(p.values)

function (p::MooncakePrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function AbstractPPL.prepare(adtype::AutoMooncake, problem, values::NamedTuple)
    evaluator = AbstractPPL.prepare(problem, values)
    f_vec = let evaluator = evaluator, values = values
        x -> evaluator(AbstractPPL.unflatten_from_vec(values, x))
    end
    cache = Mooncake.prepare_gradient_cache(evaluator, values; config=adtype.config)
    return MooncakePrepared(evaluator, f_vec, cache, values)
end

function (p::MooncakePrepared)(x::AbstractVector{<:AbstractFloat})
    dim = AbstractPPL.dimension(p)
    length(x) == dim ||
        throw(DimensionMismatch("expected vector of length $(dim), got $(length(x))"))
    return p.f_vec(x)
end

@inline function AbstractPPL.value_and_gradient(p::MooncakePrepared, values::NamedTuple)
    AbstractPPL.check_runtime_type(p.values, values)
    val, (_, grad) = Mooncake.value_and_gradient!!(p.cache, p.evaluator, values)
    return (val, grad)
end

end # module
