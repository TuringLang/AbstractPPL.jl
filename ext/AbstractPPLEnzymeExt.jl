module AbstractPPLEnzymeExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes: AutoEnzyme
using Enzyme: Enzyme

struct EnzymePrepared{E,F,P}
    evaluator::E
    f_vec::F
    gradient_buffer::Vector{Float64}
    prototype::P
    dim::Int
end

AbstractPPL.capabilities(::Type{<:EnzymePrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::EnzymePrepared) = p.dim

function (p::EnzymePrepared)(values::NamedTuple)
    return p.evaluator(values)
end

function (p::EnzymePrepared)(x::AbstractVector)
    return p.f_vec(x)
end

function AbstractPPL.prepare(
    ::AutoEnzyme, problem, prototype::NamedTuple
)
    evaluator = AbstractPPL.prepare(problem, prototype)
    x0 = AbstractPPL.flatten_to_vec(prototype)
    f_vec = let evaluator = evaluator, prototype = prototype
        x -> evaluator(AbstractPPL.unflatten_from_vec(prototype, x))
    end
    grad_buf = zeros(length(x0))
    return EnzymePrepared(evaluator, f_vec, grad_buf, prototype, length(x0))
end

function AbstractPPL.value_and_gradient(p::EnzymePrepared, values::NamedTuple)
    x = AbstractPPL.flatten_to_vec(values)
    fill!(p.gradient_buffer, 0.0)
    result = Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal),
        Enzyme.Const(p.f_vec),
        Enzyme.Active,
        Enzyme.Duplicated(x, p.gradient_buffer),
    )
    val = result[2]  # autodiff(ReverseWithPrimal, ...) returns ((adjoints...,), primal)
    grad_nt = AbstractPPL.unflatten_from_vec(p.prototype, p.gradient_buffer)
    return (val, grad_nt)
end

end # module
