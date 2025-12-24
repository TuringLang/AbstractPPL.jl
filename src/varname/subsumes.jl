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

This is done by recursively comparing each layer of the VarNames' optics.

Note that often this is not possible to determine statically, and so the results should
not be over-interpreted. In particular, `Index` optics  pose a problem. An `i::Index` will
only subsume `j::Index` if:

1. They have the same number of positional indices (`i.ix` and `j.ix`);
2. Each positional index in `i` can be determined to comprise the corresponding positional
   index in `j`; and
3. The keyword indices of `i` (`i.kw`) are a superset of those in `j.kw`).

In all other cases, `subsumes` will conservatively return `false`, even though in practice
it might well be that `i` does subsume `j`. Some examples where subsumption cannot be
determined statically are:

- Subsumption between different forms of indexing is not supported, e.g. `x[4]` and `x[2,
  2]` are not considered to subsume each other, even though they might in practice (e.g. if
  `x` is a 2x2 matrix).
- When dynamic indices (that are not equal) are present. (Dynamic indices that are equal do
  subsume each other.)
- Non-standard indices, e.g. `Not(4)`, `2..3`, etc. Again, these only subsume each other
  when they are equal.
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
    return _subsumes_positional(i.ix, j.ix) &&
           _subsumes_keyword(i.kw, j.kw) &&
           subsumes(i.child, j.child)
end

function _subsumes_positional(i::Tuple, j::Tuple)
    return (length(i) == length(j)) && all(_subsumes_index.(i, j))
end
function _subsumes_keyword(i::NamedTuple{f1}, j::NamedTuple{f2}) where {f1,f2}
    for name in f2
        if !(name in f1) || !(_subsumes_index(i[name], j[name]))
            return false
        end
    end
    return true
end

_subsumes_index(a::DynamicIndex, b::DynamicIndex) = a == b
_subsumes_index(a::DynamicIndex, ::Any) = false
_subsumes_index(::DynamicIndex, ::Colon) = false
_subsumes_index(::Colon, ::DynamicIndex) = true
_subsumes_index(::Any, ::DynamicIndex) = false
_subsumes_index(::Colon, ::Any) = true
_subsumes_index(::Colon, ::Colon) = true
_subsumes_index(::Any, ::Colon) = false
_subsumes_index(a::AbstractVector, b::Any) = issubset(b, a)
_subsumes_index(a::AbstractVector, b::Colon) = false
_subsumes_index(a::AbstractVector, b::DynamicIndex) = false
_subsumes_index(a::Any, b::Any) = a == b
