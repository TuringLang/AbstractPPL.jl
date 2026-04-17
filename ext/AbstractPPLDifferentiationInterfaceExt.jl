module AbstractPPLDifferentiationInterfaceExt

using AbstractPPL: AbstractPPL, DerivativeOrder
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI

struct DIPrepared{E,B,C}
    evaluator::E
    backend::B
    prep::C
    dim::Int
end

AbstractPPL.capabilities(::Type{<:DIPrepared}) = DerivativeOrder{1}()
AbstractPPL.dimension(p::DIPrepared) = p.dim

function (p::DIPrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(p.dim), but got length $(length(x))."
        ),
    )
    return p.evaluator(x)
end

# This extension handles the generic `AbstractADType` vector path directly so
# DifferentiationInterface backends can opt in without a fallback method in
# `src/ADProblems.jl` forcing a precompile-time method overwrite.
function AbstractPPL.prepare(
    adtype::ADTypes.AbstractADType, problem, x::AbstractVector{<:AbstractFloat}
)
    evaluator = AbstractPPL.prepare(problem, x)
    prep = DI.prepare_gradient(evaluator, adtype, x)
    return DIPrepared(evaluator, adtype, prep, length(x))
end

@inline function AbstractPPL.value_and_gradient(
    p::DIPrepared, x::AbstractVector{<:AbstractFloat}
)
    length(x) == p.dim || throw(
        DimensionMismatch(
            "Expected a vector of length $(p.dim), but got length $(length(x))."
        ),
    )
    return DI.value_and_gradient(p.evaluator, p.prep, p.backend, x)
end

end # module
