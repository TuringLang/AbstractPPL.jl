module AbstractPPLDistributionsExt

using AbstractPPL: AbstractPPL, VarName, Accessors
using Distributions: Distributions

# TODO(penelopeysm): Figure out tuple / namedtuple distributions, and LKJCholesky (grr)
function AbstractPPL.hasvalue(
    vals::AbstractDict, vn::VarName, dist::Distributions.Distribution
)
    @warn "`hasvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `hasvalue(vals, vn)`."
    return AbstractPPL.hasvalue(vals, vn)
end
function AbstractPPL.hasvalue(
    vals::AbstractDict, vn::VarName, ::Distributions.UnivariateDistribution
)
    return AbstractPPL.hasvalue(vals, vn)
end
function AbstractPPL.hasvalue(
    vals::AbstractDict{<:VarName},
    vn::VarName{sym},
    dist::Union{Distributions.MultivariateDistribution,Distributions.MatrixDistribution},
) where {sym}
    # If `vn` is present as-is, then we are good
    AbstractPPL.hasvalue(vals, vn) && return true
    # If not, then we need to check inside `vals` to see if a subset of
    # `vals` is enough to reconstruct `vn`. For example, if `vals` contains
    # `x[1]` and `x[2]`, and `dist` is `MvNormal(zeros(2), I)`, then we
    # can reconstruct `x`. If `dist` is `MvNormal(zeros(3), I)`, then we
    # can't.
    # To do this, we get the size of the distribution and iterate over all
    # possible indices. If every index can be found in `subsumed_keys`, then we
    # can return true.
    sz = size(dist)
    for idx in Iterators.product(map(Base.OneTo, sz)...)
        new_optic = if AbstractPPL.getoptic(vn) === identity
            Accessors.IndexLens(idx)
        else
            Accessors.IndexLens(idx) ∘ AbstractPPL.getoptic(vn)
        end
        new_vn = VarName{sym}(new_optic)
        AbstractPPL.hasvalue(vals, new_vn) || return false
    end
    return true
end

end
