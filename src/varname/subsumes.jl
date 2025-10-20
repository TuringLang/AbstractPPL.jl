"""
    inspace(vn::Union{VarName, Symbol}, space::Tuple)

Check whether `vn`'s variable symbol is in `space`.  The empty tuple counts as the "universal space"
containing all variables. Subsumption (see [`subsumes`](@ref)) is respected.

## Examples

```jldoctest
julia> inspace(@varname(x[1][2:3]), ())
true

julia> inspace(@varname(x[1][2:3]), (:x,))
true

julia> inspace(@varname(x[1][2:3]), (@varname(x),))
true

julia> inspace(@varname(x[1][2:3]), (@varname(x[1:10]), :y))
true

julia> inspace(@varname(x[1][2:3]), (@varname(x[:][2:4]), :y))
true

julia> inspace(@varname(x[1][2:3]), (@varname(x[1:10]),))
true
```
"""
inspace(vn, space::Tuple{}) = true # empty tuple is treated as universal space
inspace(vn, space::Tuple) = vn in space
inspace(vn::VarName, space::Tuple{}) = true
inspace(vn::VarName, space::Tuple) = any(_in(vn, s) for s in space)

_in(vn::VarName, s::Symbol) = getsym(vn) == s
_in(vn::VarName, s::VarName) = subsumes(s, vn)

"""
    subsumes(u::VarName, v::VarName)

Check whether the variable name `v` describes a sub-range of the variable `u`.  Supported
indexing:

  - Scalar:

  ```jldoctest
  julia> subsumes(@varname(x), @varname(x[1, 2]))
  true
  
  julia> subsumes(@varname(x[1, 2]), @varname(x[1, 2][3]))
  true
  ```
  
  - Array of scalar: basically everything that fulfills `issubset`.
  
  ```jldoctest
  julia> subsumes(@varname(x[[1, 2], 3]), @varname(x[1, 3]))
  true
  
  julia> subsumes(@varname(x[1:3]), @varname(x[2][1]))
  true
  ```
  
  - Slices:
  
  ```jldoctest
  julia> subsumes(@varname(x[2, :]), @varname(x[2, 10][1]))
  true
  ```

Currently _not_ supported are: 

  - Boolean indexing, literal `CartesianIndex` (these could be added, though)
  - Linear indexing of multidimensional arrays: `x[4]` does not subsume `x[2, 2]` for a matrix `x`
  - Trailing ones: `x[2, 1]` does not subsume `x[2]` for a vector `x`
"""
function subsumes(u::VarName, v::VarName)
    return getsym(u) == getsym(v) && subsumes(getoptic(u), getoptic(v))
end

# Idea behind `subsumes` for `Lens` is that we traverse the two lenses in parallel,
# checking `subsumes` for every level. This for example means that if we are comparing
# `PropertyLens{:a}` and `PropertyLens{:b}` we immediately know that they do not subsume
# each other since at the same level/depth they access different properties.
# E.g. `x`, `x[1]`, i.e. `u` is always subsumed by `t`
subsumes(::typeof(identity), ::typeof(identity)) = true
subsumes(::typeof(identity), ::ALLOWED_OPTICS) = true
subsumes(::ALLOWED_OPTICS, ::typeof(identity)) = false

function subsumes(t::ComposedFunction, u::ComposedFunction)
    return subsumes(t.outer, u.outer) && subsumes(t.inner, u.inner)
end

# If `t` is still a composed lens, then there is no way it can subsume `u` since `u` is a
# leaf of the "lens-tree".
subsumes(t::ComposedFunction, u::PropertyLens) = false
# Here we need to check if `u.inner` (i.e. the next lens to be applied from `u`) is
# subsumed by `t`, since this would mean that the rest of the composition is also subsumed
# by `t`.
subsumes(t::PropertyLens, u::ComposedFunction) = subsumes(t, u.inner)

# For `PropertyLens` either they have the same `name` and thus they are indeed the same.
subsumes(t::PropertyLens{name}, u::PropertyLens{name}) where {name} = true
# Otherwise they represent different properties, and thus are not the same.
subsumes(t::PropertyLens, u::PropertyLens) = false

# PropertyLens and IndexLens can't subsume each other
subsumes(::PropertyLens, ::IndexLens) = false
subsumes(::IndexLens, ::PropertyLens) = false

# Indices subsumes if they are subindices, i.e. we just call `_issubindex`.
# FIXME: Does not support `DynamicIndexLens`.
# FIXME: Does not correctly handle cases such as `subsumes(x, x[:])`
#        (but neither did old implementation).
function subsumes(
    t::Union{IndexLens,ComposedFunction{<:ALLOWED_OPTICS,<:IndexLens}},
    u::Union{IndexLens,ComposedFunction{<:ALLOWED_OPTICS,<:IndexLens}},
)
    return subsumes_indices(t, u)
end

"""
    subsumedby(t, u)

True if `t` is subsumed by `u`, i.e., if `subsumes(u, t)` is true.
"""
subsumedby(t, u) = subsumes(u, t)
uncomparable(t, u) = t ⋢ u && u ⋢ t
const ⊒ = subsumes
const ⊑ = subsumedby
const ⋣ = !subsumes
const ⋢ = !subsumedby
const ≍ = uncomparable

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
subsumes_index(i, j) = i == j
