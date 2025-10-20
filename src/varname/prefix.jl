### Functionality for prefixing and unprefixing VarNames.

"""
    optic_to_vn(optic)

Convert an Accessors optic to a VarName. This is best explained through
examples.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL.optic_to_vn(Accessors.@o _.a)
a

julia> AbstractPPL.optic_to_vn(Accessors.@o _.a.b)
a.b

julia> AbstractPPL.optic_to_vn(Accessors.@o _.a[1])
a[1]
```

The outermost layer of the optic (technically, what Accessors.jl calls the
'innermost') must be a `PropertyLens`, or else it will fail. This is because a
VarName needs to have a symbol.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL.optic_to_vn(Accessors.@o _[1])
ERROR: ArgumentError: optic_to_vn: could not convert optic `(@o _[1])` to a VarName
[...]
```
"""
function optic_to_vn(::Accessors.PropertyLens{sym}) where {sym}
    return VarName{sym}()
end
function optic_to_vn(
    o::ComposedFunction{Outer,Accessors.PropertyLens{sym}}
) where {Outer,sym}
    return VarName{sym}(o.outer)
end
optic_to_vn(o::ComposedFunction) = optic_to_vn(normalise(o))
function optic_to_vn(@nospecialize(o))
    msg = "optic_to_vn: could not convert optic `$o` to a VarName"
    throw(ArgumentError(msg))
end

unprefix_optic(o, ::typeof(identity)) = o  # Base case
function unprefix_optic(optic, optic_prefix)
    # Technically `unprefix_optic` only receives optics that were part of
    # VarNames, so the optics should already be normalised (in the inner
    # constructor of the VarName). However I guess it doesn't hurt to do it
    # again to be safe.
    optic = normalise(optic)
    optic_prefix = normalise(optic_prefix)
    # strip one layer of the optic and check for equality
    head = _head(optic)
    head_prefix = _head(optic_prefix)
    if head != head_prefix
        msg = "could not remove prefix $(optic_prefix) from optic $(optic)"
        throw(ArgumentError(msg))
    end
    # recurse
    return unprefix_optic(_tail(optic), _tail(optic_prefix))
end

"""
    unprefix(vn::VarName, prefix::VarName)

Remove a prefix from a VarName.

```jldoctest
julia> AbstractPPL.unprefix(@varname(y.x), @varname(y))
x

julia> AbstractPPL.unprefix(@varname(y.x.a), @varname(y))
x.a

julia> AbstractPPL.unprefix(@varname(y[1].x), @varname(y[1]))
x

julia> AbstractPPL.unprefix(@varname(y), @varname(n))
ERROR: ArgumentError: could not remove prefix n from VarName y
[...]
```
"""
function unprefix(
    vn::VarName{sym_vn}, prefix::VarName{sym_prefix}
) where {sym_vn,sym_prefix}
    if sym_vn != sym_prefix
        msg = "could not remove prefix $(prefix) from VarName $(vn)"
        throw(ArgumentError(msg))
    end
    optic_vn = getoptic(vn)
    optic_prefix = getoptic(prefix)
    return optic_to_vn(unprefix_optic(optic_vn, optic_prefix))
end

"""
    prefix(vn::VarName, prefix::VarName)

Add a prefix to a VarName.

```jldoctest
julia> AbstractPPL.prefix(@varname(x), @varname(y))
y.x

julia> AbstractPPL.prefix(@varname(x.a), @varname(y))
y.x.a

julia> AbstractPPL.prefix(@varname(x.a), @varname(y[1]))
y[1].x.a
```
"""
function prefix(vn::VarName{sym_vn}, prefix::VarName{sym_prefix}) where {sym_vn,sym_prefix}
    optic_vn = getoptic(vn)
    optic_prefix = getoptic(prefix)
    new_optic_vn = optic_vn ∘ PropertyLens{sym_vn}() ∘ optic_prefix
    return VarName{sym_prefix}(new_optic_vn)
end
