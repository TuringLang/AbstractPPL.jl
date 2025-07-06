module AbstractPPLDistributionsExt

using AbstractPPL: AbstractPPL, VarName, Accessors
using Distributions: Distributions
using LinearAlgebra: Cholesky, LowerTriangular, UpperTriangular

#=
This section is copied from Accessors.jl's documentation:
https://juliaobjects.github.io/Accessors.jl/stable/examples/custom_macros/

It defines a wrapper that, when called with `set`, mutates the original value
rather than returning a new value. We need this because the non-mutating optics
don't work for triangular matrices (and hence LKJCholesky): see
https://github.com/JuliaObjects/Accessors.jl/issues/203
=#
struct Lens!{L}
    pure::L
end
(l::Lens!)(o) = l.pure(o)
function Accessors.set(o, l::Lens!{<:ComposedFunction}, val)
    o_inner = l.pure.inner(o)
    return Accessors.set(o_inner, Lens!(l.pure.outer), val)
end
function Accessors.set(o, l::Lens!{Accessors.PropertyLens{prop}}, val) where {prop}
    setproperty!(o, prop, val)
    return o
end
function Accessors.set(o, l::Lens!{<:Accessors.IndexLens}, val)
    o[l.pure.indices...] = val
    return o
end

"""
    get_optics(dist::MultivariateDistribution)
    get_optics(dist::MatrixDistribution)
    get_optics(dist::LKJCholesky)

Return a complete set of optics for each element of the type returned by `rand(dist)`.
"""
function get_optics(
    dist::Union{Distributions.MultivariateDistribution,Distributions.MatrixDistribution}
)
    indices = CartesianIndices(size(dist))
    return map(idx -> Accessors.IndexLens(idx.I), indices)
end
function get_optics(dist::Distributions.LKJCholesky)
    is_up = dist.uplo == 'U'
    cartesian_indices = filter(CartesianIndices(size(dist))) do cartesian_index
        i, j = cartesian_index.I
        is_up ? i <= j : i >= j
    end
    # there is an additional layer as we need to access `.L` or `.U` before we
    # can index into it
    field_lens = is_up ? (Accessors.@o _.U) : (Accessors.@o _.L)
    return map(idx -> Accessors.IndexLens(idx.I) ∘ field_lens, cartesian_indices)
end

"""
    make_empty_value(dist::MultivariateDistribution)
    make_empty_value(dist::MatrixDistribution)
    make_empty_value(dist::LKJCholesky)

Construct a fresh value filled with zeros that corresponds to the size of `dist`.

For all distributions that this function accepts, it should hold that
`o(make_empty_value(dist))` is zero for all `o` in `get_optics(dist)`.
"""
function make_empty_value(
    dist::Union{Distributions.MultivariateDistribution,Distributions.MatrixDistribution}
)
    return zeros(size(dist))
end
function make_empty_value(dist::Distributions.LKJCholesky)
    if dist.uplo == 'U'
        return Cholesky(UpperTriangular(zeros(size(dist))))
    else
        return Cholesky(LowerTriangular(zeros(size(dist))))
    end
end

"""
    hasvalue(
        vals::AbstractDict,
        vn::VarName,
        dist::Distribution;
        error_on_incomplete::Bool=false
    )

Check if `vals` contains values for `vn` that is compatible with the
distribution `dist`.

This is a more general version of `hasvalue(vals, vn)`, in that even if
`vn` itself is not inside `vals`, it further checks if `vals` contains
sub-values of `vn` that can be used to reconstruct `vn` given `dist`.

The `error_on_incomplete` flag can be used to detect cases where _some_ of
the values needed for `vn` are present, but others are not. This may help
to detect invalid cases where the user has provided e.g. data of the wrong
shape.

For example:

```jldoctest; setup=:(using Distributions, LinearAlgebra))
julia> d = Dict(@varname(x[1]) => 1.0, @varname(x[2]) => 2.0);

julia> hasvalue(d, @varname(x), MvNormal(zeros(2), I))
true

julia> hasvalue(d, @varname(x), MvNormal(zeros(3), I))
false

julia> hasvalue(d, @varname(x), MvNormal(zeros(3), I); error_on_incomplete=true)
ERROR: hasvalue: only partial values for `x` found in the values provided
[...]
```
"""
function AbstractPPL.hasvalue(
    vals::AbstractDict,
    vn::VarName,
    dist::Distributions.Distribution;
    error_on_incomplete::Bool=false,
)
    @warn "`hasvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `hasvalue(vals, vn)`."
    return AbstractPPL.hasvalue(vals, vn)
end
function AbstractPPL.hasvalue(
    vals::AbstractDict,
    vn::VarName,
    ::Distributions.UnivariateDistribution;
    error_on_incomplete::Bool=false,
)
    # TODO(penelopeysm): We could also implement a check for the type to catch
    # invalid values. Unsure if that is worth it. It may be easier to just let
    # the user handle it.
    return AbstractPPL.hasvalue(vals, vn)
end
function AbstractPPL.hasvalue(
    vals::AbstractDict{<:VarName},
    vn::VarName{sym},
    dist::Union{
        Distributions.MultivariateDistribution,
        Distributions.MatrixDistribution,
        Distributions.LKJCholesky,
    };
    error_on_incomplete::Bool=false,
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
    optics = get_optics(dist)
    original_optic = AbstractPPL.getoptic(vn)
    expected_vns = map(o -> VarName{sym}(o ∘ original_optic), optics)
    if all(sub_vn -> AbstractPPL.hasvalue(vals, sub_vn), expected_vns)
        return true
    else
        if error_on_incomplete &&
            any(sub_vn -> AbstractPPL.hasvalue(vals, sub_vn), expected_vns)
            error("hasvalue: only partial values for `$vn` found in the values provided")
        end
        return false
    end
end

"""
    getvalue(vals::AbstractDict, vn::VarName, dist::Distribution)

Retrieve the value of `vn` from `vals`, using the distribution `dist` to
reconstruct the value if necessary.

This is a more general version of `getvalue(vals, vn)`, in that even if `vn`
itself is not inside `vals`, it can still reconstruct the value of `vn`
from sub-values of `vn` that are present in `vals`.

For example:

```jldoctest; setup=:(using Distributions, LinearAlgebra))
julia> d = Dict(@varname(x[1]) => 1.0, @varname(x[2]) => 2.0);

julia> getvalue(d, @varname(x), MvNormal(zeros(2), I))
2-element Vector{Float64}:
 1.0
 2.0

julia> # Use `hasvalue` to check for this case before calling `getvalue`.
       getvalue(d, @varname(x), MvNormal(zeros(3), I))
ERROR: getvalue: `x` was not found in the values provided
[...]
```
"""
function AbstractPPL.getvalue(
    vals::AbstractDict, vn::VarName, dist::Distributions.Distribution;
)
    @warn "`getvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `getvalue(vals, vn)`."
    return AbstractPPL.getvalue(vals, vn)
end
function AbstractPPL.getvalue(
    vals::AbstractDict, vn::VarName, ::Distributions.UnivariateDistribution;
)
    # TODO(penelopeysm): We could also implement a check for the type to catch
    # invalid values. Unsure if that is worth it. It may be easier to just let
    # the user handle it.
    return AbstractPPL.getvalue(vals, vn)
end
function AbstractPPL.getvalue(
    vals::AbstractDict{<:VarName},
    vn::VarName{sym},
    dist::Union{
        Distributions.MultivariateDistribution,
        Distributions.MatrixDistribution,
        Distributions.LKJCholesky,
    };
) where {sym}
    # If `vn` is present as-is, then we can just return that
    AbstractPPL.hasvalue(vals, vn) && return AbstractPPL.getvalue(vals, vn)
    # If not, then we need to start looking inside `vals`, in exactly the
    # same way we did for `hasvalue`.
    optics = get_optics(dist)
    original_optic = AbstractPPL.getoptic(vn)
    expected_vns = map(o -> VarName{sym}(o ∘ original_optic), optics)
    if all(sub_vn -> AbstractPPL.hasvalue(vals, sub_vn), expected_vns)
        # Reconstruct the value index by index.
        value = make_empty_value(dist)
        for (o, sub_vn) in zip(optics, expected_vns)
            # Retrieve the value of this given index
            sub_value = AbstractPPL.getvalue(vals, sub_vn)
            # Set it inside the value we're reconstructing.
            # Note: `o` is normally non-mutating. We have to wrap it in `Lens!`
            # to make it mutating, because Cholesky distributions are broken
            # by https://github.com/JuliaObjects/Accessors.jl/issues/203.
            Accessors.set(value, Lens!(o), sub_value)
        end
        return value
    else
        error("getvalue: $(vn) was not found in the values provided")
    end
end

end
