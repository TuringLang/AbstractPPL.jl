"""
    subsumes(parent::VarName, child::VarName)

Check whether the variable name `child` describes a sub-range of the variable `parent`,
i.e., is contained within it.

```jldoctest
julia> subsumes(@varname(x), @varname(x[1, 2]))
true

julia> subsumes(@varname(x[1, 2]), @varname(x[1, 2][3]))
true
```

Note that often this is not possible to determine statically. For example:

- When dynamic indices are present, subsumption cannot be determined, unless `child ==
  parent`.
- Subsumption between different forms of indexing is not supported, e.g. `x[4]` and `x[2,
  2]` are not considered to subsume each other, even though they might in practice (e.g. if
  `x` is a 2x2 matrix).

In such cases, `subsumes` will conservatively return `false`.
"""
function subsumes(u::VarName, v::VarName)
    return getsym(u) == getsym(v) && subsumes(getoptic(u), getoptic(v))
end
subsumes(::Iden, ::Iden) = true
subsumes(::Iden, ::AbstractOptic) = true
subsumes(::AbstractOptic, ::Iden) = false
subsumes(t::Property{name}, u::Property{name}) where {name} = subsumes(t.child, u.child)
subsumes(t::Property, u::Property) = false
subsumes(::Property, ::Index) = false
subsumes(::Index, ::Property) = false

function subsumes(i::Index, j::Index)
    # TODO(penelopeysm): What we really want to do is to zip i.ix and j.ix
    # and check that each index in `i.ix` subsumes the corresponding
    # entry in `j.ix`. If that is true, then we can continue recursing.
    return if i.ix == j.ix && i.kw == j.kw
        subsumes(i.child, j.child)
    else
        error("Not implemented.")
    end
end

#=
# Since expressions such as `x[:][:][:][1]` and `x[1]` are equal,
# the indexing behavior must be considered jointly.
# Therefore we must recurse until we reach something that is NOT
# indexing, and then consider the sequence of indices leading up to this.
"""
    subsumes_indices(t, u)

Return `true` if the indexing represented by `t` subsumes `u`.

This is mostly useful for comparing compositions involving `IndexLens`
e.g. `_[1][2].a[2]` and `_[1][2].a`. In such a scenario we do the following:
1. Combine `[1][2]` into a `Tuple` of indices using [`combine_indices`](@ref).
2. Do the same for `[1][2]`.
3. Compare the two tuples from (1) and (2) using `subsumes_indices`.
4. Since we're still undecided, we call `subsume(@o(_.a[2]), @o(_.a))`
   which then returns `false`.

# Example
```jldoctest; setup=:(using Accessors; using AbstractPPL: subsumes_indices)
julia> t = @o(_[1].a); u = @o(_[1]);

julia> subsumes_indices(t, u)
false

julia> subsumes_indices(u, t)
true

julia> # `identity` subsumes all.
       subsumes_indices(identity, t)
true

julia> # None subsumes `identity`.
       subsumes_indices(t, identity)
false

julia> AbstractPPL.subsumes(@o(_[1][2].a[2]), @o(_[1][2].a))
false

julia> AbstractPPL.subsumes(@o(_[1][2].a), @o(_[1][2].a[2]))
true
```
"""
function subsumes_indices(t::ALLOWED_OPTICS, u::ALLOWED_OPTICS)
    t_indices, t_next = combine_indices(t)
    u_indices, u_next = combine_indices(u)

    # If we already know that `u` is not subsumed by `t`, return early.
    if !subsumes_indices(t_indices, u_indices)
        return false
    end

    if t_next === nothing
        # Means that there's nothing left for `t` and either nothing
        # or something left for `u`, i.e. `t` indeed `subsumes` `u`.
        return true
    elseif u_next === nothing
        # If `t_next` is not `nothing` but `u_next` is, then
        # `t` does not subsume `u`.
        return false
    end

    # If neither is `nothing` we continue.
    return subsumes(t_next, u_next)
end

"""
    combine_indices(optic)

Return sequential indexing into a single `Tuple` of indices,
e.g. `x[:][1][2]` becomes `((Colon(), ), (1, ), (2, ))`.

The result is compatible with [`subsumes_indices`](@ref) for `Tuple` input.
"""
combine_indices(optic::ALLOWED_OPTICS) = (), optic
combine_indices(optic::IndexLens) = (optic.indices,), nothing
function combine_indices(optic::ComposedFunction{<:ALLOWED_OPTICS,<:IndexLens})
    indices, next = combine_indices(optic.outer)
    return (optic.inner.indices, indices...), next
end

"""
    subsumes_indices(left_indices::Tuple, right_indices::Tuple)

Return `true` if `right_indices` is subsumed by `left_indices`.  `left_indices` is assumed to be 
concretized and consist of either `Int`s or `AbstractArray`s of scalar indices that are supported 
by array A.

Currently _not_ supported are: 
- Boolean indexing, literal `CartesianIndex` (these could be added, though)
- Linear indexing of multidimensional arrays: `x[4]` does not subsume `x[2, 2]` for a matrix `x`
- Trailing ones: `x[2, 1]` does not subsume `x[2]` for a vector `x`
"""
subsumes_indices(::Tuple{}, ::Tuple{}) = true  # x subsumes x
subsumes_indices(::Tuple{}, ::Tuple) = true    # x subsumes x...
subsumes_indices(::Tuple, ::Tuple{}) = false   # x... does not subsume x
function subsumes_indices(t1::Tuple, t2::Tuple)  # does x[i]... subsume x[j]...?
    first_subsumed = all(Base.splat(subsumes_index), zip(first(t1), first(t2)))
    return first_subsumed && subsumes_indices(Base.tail(t1), Base.tail(t2))
end

subsumes_index(i::Colon, ::Colon) = error("Colons cannot be subsumed")
subsumes_index(i, ::Colon) = error("Colons cannot be subsumed")
# Necessary to avoid ambiguity errors.
subsumes_index(::AbstractVector, ::Colon) = error("Colons cannot be subsumed")
subsumes_index(i::Colon, j) = true
subsumes_index(i::AbstractVector, j) = issubset(j, i)
subsumes_index(i, j) = i == j =#
