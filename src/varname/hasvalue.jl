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
ERROR: x[2] was not found in the NamedTuple provided
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
ERROR: x[2] was not found in the dictionary provided
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
ERROR: x[1][2] was not found in the dictionary provided
[...]

julia> getvalue(vals, @varname(x[2][1]))
ERROR: x[2][1] was not found in the dictionary provided
[...]

julia> getvalue(vals, @varname(x))
ERROR: x was not found in the dictionary provided
[...]
```

Dictionaries can present ambiguous cases where the same variable is specified
twice at different levels. In such a situation, `getvalue` attempts to find an
exact match, and if that fails it returns the value with the most specific key.

!!! note
    It is the user's responsibility to avoid such cases by ensuring that the
    dictionary passed in does not contain the same value specified multiple
    times.

```jldoctest
julia> vals = Dict(@varname(x) => [[1.0]], @varname(x[1]) => [2.0]);

julia> # Here, the `x[1]` key is not used because `x` is an exact match.
       getvalue(vals, @varname(x))
1-element Vector{Vector{Float64}}:
 [1.0]

julia> # Likewise, the `x` key is not used because `x[1]` is an exact match.
       getvalue(vals, @varname(x[1]))
1-element Vector{Float64}:
 2.0

julia> # No exact match, so the most specific key, i.e. `x[1]`, is used.
       getvalue(vals, @varname(x[1][1]))
2.0
```
"""
function getvalue(vals::NamedTuple, vn::VarName{sym}) where {sym}
    optic = getoptic(vn)
    if haskey(vals, sym) && canview(optic, getproperty(vals, sym))
        return optic(vals[sym])
    else
        error("$(vn) was not found in the NamedTuple provided")
    end
end

# For the Dict case, it is more complicated. There are two cases:
# 1. `vn` itself is already a key of `vals` (the easy case)
# 2. `vn` is not a key of `vals`, but some parent of `vn` is a key of `vals`
#    (the harder case). For example, if `vn` is `x[1][2]`, then we need to
#    check if either `x` or `x[1]` is a key of `vals`, and if so, whether
#    we can index into the corresponding value.
function getvalue(vals::AbstractDict{<:VarName}, vn::VarName{sym}) where {sym}
    # First we check if `vn` is present as is.
    haskey(vals, vn) && return vals[vn]

    # Otherwise, we start by testing the `vn` one level up (e.g., if `vn` is
    # `x[1][2]`, we start by checking if `x[1]` is present, then `x`). We will
    # then keep removing optics from `test_optic`, either until we find a key
    # that is present, or until we run out of optics to test (which happens 
    # after getoptic(test_vn) == identity).
    o = getoptic(vn)
    test_vn = VarName{sym}(_init(o))
    test_optic = _last(o)

    while true
        if haskey(vals, test_vn) && canview(test_optic, vals[test_vn])
            return test_optic(vals[test_vn])
        else
            # Try to move the outermost optic from test_vn into test_optic.
            # If test_vn is already an identity, we can't, so we stop.
            o = getoptic(test_vn)
            o == identity && error("$(vn) was not found in the dictionary provided")
            test_vn = VarName{sym}(_init(o))
            test_optic = normalise(test_optic ∘ _last(o))
        end
    end
end

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
function hasvalue(vals::AbstractDict{<:VarName}, vn::VarName{sym}) where {sym}
    # First we check if `vn` is present as is.
    haskey(vals, vn) && return true

    # Otherwise, we start by testing the `vn` one level up (e.g., if `vn` is
    # `x[1][2]`, we start by checking if `x[1]` is present, then `x`). We will
    # then keep removing optics from `test_optic`, either until we find a key
    # that is present, or until we run out of optics to test (which happens 
    # after getoptic(test_vn) == identity).
    o = getoptic(vn)
    test_vn = VarName{sym}(_init(o))
    test_optic = _last(o)

    while true
        if haskey(vals, test_vn) && canview(test_optic, vals[test_vn])
            return true
        else
            # Try to move the outermost optic from test_vn into test_optic.
            # If test_vn is already an identity, we can't, so we stop.
            o = getoptic(test_vn)
            o == identity && return false
            test_vn = VarName{sym}(_init(o))
            test_optic = normalise(test_optic ∘ _last(o))
        end
    end
    return false
end
