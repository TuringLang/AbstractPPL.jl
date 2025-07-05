"""
    canview(optic, container)

Return `true` if `optic` can be used to view `container`, and `false` otherwise.

# Examples
```jldoctest; setup=:(using Accessors)
julia> AbstractPPL.canview(@o(_.a), (a = 1.0, ))
true

julia> AbstractPPL.canview(@o(_.a), (b = 1.0, )) # property `a` does not exist
false

julia> AbstractPPL.canview(@o(_.a[1]), (a = [1.0, 2.0], ))
true

julia> AbstractPPL.canview(@o(_.a[3]), (a = [1.0, 2.0], )) # out of bounds
false
```
"""
canview(optic, container) = false
canview(::typeof(identity), _) = true
function canview(::Accessors.PropertyLens{field}, x) where {field}
    return hasproperty(x, field)
end

# `IndexLens`: only relevant if `x` supports indexing.
canview(optic::Accessors.IndexLens, x) = false
function canview(optic::Accessors.IndexLens, x::AbstractArray)
    return checkbounds(Bool, x, optic.indices...)
end

# `ComposedFunction`: check that we can view `.inner` and `.outer`, but using
# value extracted using `.inner`.
function canview(optic::ComposedFunction, x)
    return canview(optic.inner, x) && canview(optic.outer, optic.inner(x))
end

"""
    getvalue(vals::NamedTuple, vn::VarName)
    getvalue(vals::AbstractDict{<:VarName}, vn::VarName)

Return the value(s) in `vals` represented by `vn`.

# Examples

For `NamedTuple`:

```jldoctest
julia> vals = (x = [1.0],);

julia> getvalue(vals, @varname(x)) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> getvalue(vals, @varname(x[1])) # different from `getindex`
1.0

julia> getvalue(vals, @varname(x[2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]
```

For `AbstractDict`:

```jldoctest
julia> vals = Dict(@varname(x) => [1.0]);

julia> getvalue(vals, @varname(x)) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> getvalue(vals, @varname(x[1])) # different from `getindex`
1.0

julia> getvalue(vals, @varname(x[2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]
```

In the `AbstractDict` case we can also have keys such as `v[1]`:

```jldoctest
julia> vals = Dict(@varname(x[1]) => [1.0,]);

julia> getvalue(vals, @varname(x[1])) # same as `getindex`
1-element Vector{Float64}:
 1.0

julia> getvalue(vals, @varname(x[1][1])) # different from `getindex`
1.0

julia> getvalue(vals, @varname(x[1][2]))
ERROR: BoundsError: attempt to access 1-element Vector{Float64} at index [2]
[...]

julia> getvalue(vals, @varname(x[2][1]))
ERROR: KeyError: key x[2][1] not found
[...]
```
"""
getvalue(vals::NamedTuple, vn::VarName) = get(vals, vn)
getvalue(vals::AbstractDict, vn::VarName) = nested_getindex(vals, vn)

"""
    hasvalue(vals::NamedTuple, vn::VarName)
    hasvalue(vals::AbstractDict{<:VarName}, vn::VarName)

Determine whether `vals` contains a value for a given `vn`.

# Examples
With `x` as a `NamedTuple`:

```jldoctest
julia> hasvalue((x = 1.0, ), @varname(x))
true

julia> hasvalue((x = 1.0, ), @varname(x[1]))
false

julia> hasvalue((x = [1.0],), @varname(x))
true

julia> hasvalue((x = [1.0],), @varname(x[1]))
true

julia> hasvalue((x = [1.0],), @varname(x[2]))
false
```

With `x` as a `AbstractDict`:

```jldoctest
julia> hasvalue(Dict(@varname(x) => 1.0, ), @varname(x))
true

julia> hasvalue(Dict(@varname(x) => 1.0, ), @varname(x[1]))
false

julia> hasvalue(Dict(@varname(x) => [1.0]), @varname(x))
true

julia> hasvalue(Dict(@varname(x) => [1.0]), @varname(x[1]))
true

julia> hasvalue(Dict(@varname(x) => [1.0]), @varname(x[2]))
false
```

In the `AbstractDict` case we can also have keys such as `v[1]`:

```jldoctest
julia> vals = Dict(@varname(x[1]) => [1.0,]);

julia> hasvalue(vals, @varname(x[1])) # same as `haskey`
true

julia> hasvalue(vals, @varname(x[1][1])) # different from `haskey`
true

julia> hasvalue(vals, @varname(x[1][2]))
false

julia> hasvalue(vals, @varname(x[2][1]))
false
```
"""
function hasvalue(vals::NamedTuple, vn::VarName{sym}) where {sym}
    return haskey(vals, sym) && canview(getoptic(vn), getproperty(vals, sym))
end

# For the Dict case, it is more complicated. There are two cases:
# 1. `vn` itself is already a key of `vals` (the easy case)
# 2. `vn` is not a key of `vals`, but some parent of `vn` is a key of `vals`
#    (the harder case). For example, if `vn` is `x[1][2]`, then we need to
#    check if either `x` or `x[1]` is a key of `vals`, and if so, whether
#    we can index into the corresponding value.
function hasvalue(vals::AbstractDict{<:VarName}, vn::VarName{sym}) where {sym}
    # First we check if `vn` is present as is.
    haskey(vals, vn) && return true

    # Otherwise, we start by testing the bare `vn` (e.g., if `vn` is `x[1][2]`,
    # we start by checking if `x` is present). We will then keep adding optics
    # to `test_optic`, either until we find a key that is present, or until we
    # run out of optics to test (which is determined by _inner(test_optic) ==
    # identity).
    test_vn = VarName{sym}()
    test_optic = getoptic(vn)
    
    while _inner(test_optic) != identity
        @show test_vn, test_optic
        if haskey(vals, test_vn)
            @show canview(test_optic, vals[test_vn])
        end
        if haskey(vals, test_vn) && canview(test_optic, vals[test_vn])
            return true
        else
            # Move the innermost optic into test_vn
            test_optic_outer = _outer(test_optic)
            test_optic_inner = _inner(test_optic)
            test_vn = VarName{sym}(test_optic_inner ∘ getoptic(test_vn))
            test_optic = test_optic_outer
        end
    end
    return false
end
# TODO(penelopeysm): Figure out tuple / namedtuple distributions, and LKJCholesky (grr)
# function hasvalue(vals::AbstractDict, vn::VarName, dist::Distribution)
#     @warn "`hasvalue(vals, vn, dist)` is not implemented for $(typeof(dist)); falling back to `hasvalue(vals, vn)`."
#     return hasvalue(vals, vn)
# end
# hasvalue(vals::AbstractDict, vn::VarName, ::UnivariateDistribution) = hasvalue(vals, vn)
# function hasvalue(
#     vals::AbstractDict{<:VarName},
#     vn::VarName{sym},
#     dist::Union{MultivariateDistribution,MatrixDistribution},
# ) where {sym}
#     # If `vn` is present as-is, then we are good
#     hasvalue(vals, vn) && return true
#     # If not, then we need to check inside `vals` to see if a subset of
#     # `vals` is enough to reconstruct `vn`. For example, if `vals` contains
#     # `x[1]` and `x[2]`, and `dist` is `MvNormal(zeros(2), I)`, then we
#     # can reconstruct `x`. If `dist` is `MvNormal(zeros(3), I)`, then we
#     # can't.
#     # To do this, we get the size of the distribution and iterate over all
#     # possible indices. If every index can be found in `subsumed_keys`, then we
#     # can return true.
#     sz = size(dist)
#     for idx in Iterators.product(map(Base.OneTo, sz)...)
#         new_optic = if getoptic(vn) === identity
#             Accessors.IndexLens(idx)
#         else
#             Accessors.IndexLens(idx) ∘ getoptic(vn)
#         end
#         new_vn = VarName{sym}(new_optic)
#         hasvalue(vals, new_vn) || return false
#     end
#     return true
# end

# """
#     nested_getindex(values::AbstractDict, vn::VarName)
#
# Return value corresponding to `vn` in `values` by also looking
# in the the actual values of the dict.
# """
# function nested_getindex(values::AbstractDict, vn::VarName)
#     maybeval = get(values, vn, nothing)
#     if maybeval !== nothing
#         return maybeval
#     end
#
#     # Split the optic into the key / `parent` and the extraction optic / `child`.
#     parent, child, issuccess = splitoptic(getoptic(vn)) do optic
#         o = optic === nothing ? identity : optic
#         haskey(values, VarName(vn, o))
#     end
#     # When combined with `VarInfo`, `nothing` is equivalent to `identity`.
#     keyoptic = parent === nothing ? identity : parent
#
#     # If we found a valid split, then we can extract the value.
#     if !issuccess
#         # At this point we just throw an error since the key could not be found.
#         throw(KeyError(vn))
#     end
#
#     # TODO: Should we also check that we `canview` the extracted `value`
#     # rather than just let it fail upon `get` call?
#     value = values[VarName(vn, keyoptic)]
#     return child(value)
# end
