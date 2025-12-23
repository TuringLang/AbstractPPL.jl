using Accessors: Accessors

"""
    AbstractOptic

An abstract type that represents the non-symbol part of a VarName, i.e., the section of the
variable that is of interest. For example, in `x.a[1][2]`, the `AbstractOptic` represents
the `.a[1][2]` part.

# Public interface

This is WIP.

- Base.show
- to_accessors(optic) -> Accessors.Lens (recovering the old representation)
- is_dynamic(optic) -> Bool (whether the optic contains any dynamic indices)
- concretize(optic, val) -> AbstractOptic (resolving any dynamic indices given the value)

Not sure if we want to introduce getters and setters and BangBang-style stuff.
"""
abstract type AbstractOptic end
function Base.show(io::IO, optic::AbstractOptic)
    print(io, "Optic(")
    _pretty_print_optic(io, optic)
    return print(io, ")")
end

"""
    Iden()

The identity optic. This is the optic used when we are referring to the entire variable.
It is also the base case for composing optics.
"""
struct Iden <: AbstractOptic end
_pretty_print_optic(::IO, ::Iden) = nothing
to_accessors(::Iden) = identity
is_dynamic(::Iden) = false
concretize(i::Iden, ::Any) = i

"""
    DynamicIndex

An abstract type representing dynamic indices such as `begin`, `end`, and `:`. These indices
are things which cannot be resolved until we provide the value that is being indexed into.
When parsing VarNames, we convert such indices into subtypes of `DynamicIndex`, and we later
mark them as requiring concretisation.
"""
abstract type DynamicIndex end
_is_dynamic_idx(::DynamicIndex) = true
_is_dynamic_idx(::Any) = false
# Fallback for all other indices
concretize(@nospecialize(ix::Any), ::Any, ::Any) = ix
_pretty_print_index(x::Any) = string(x)

struct DynamicBegin <: DynamicIndex end
concretize(::DynamicBegin, val, dim::Nothing) = Base.firstindex(val)
concretize(::DynamicBegin, val, dim) = Base.firstindex(val, dim)
_pretty_print_index(::DynamicBegin) = "begin"

struct DynamicEnd <: DynamicIndex end
concretize(::DynamicEnd, val, dim::Nothing) = Base.lastindex(val)
concretize(::DynamicEnd, val, dim) = Base.lastindex(val, dim)
_pretty_print_index(::DynamicEnd) = "end"

struct DynamicColon <: DynamicIndex end
concretize(::DynamicColon, val, dim::Nothing) = Base.firstindex(val):Base.lastindex(val)
concretize(::DynamicColon, val, dim) = Base.firstindex(val, dim):Base.lastindex(val, dim)
_pretty_print_index(::DynamicColon) = ":"

struct DynamicRange{T1,T2} <: DynamicIndex
    start::T1
    stop::T2
end
function concretize(dr::DynamicRange, axis, dim)
    start = dr.start isa DynamicIndex ? concretize(dr.start, axis, dim) : dr.start
    stop = dr.stop isa DynamicIndex ? concretize(dr.stop, axis, dim) : dr.stop
    return start:stop
end
function _pretty_print_index(dr::DynamicRange)
    return "$(_pretty_print_index(dr.start)):$(_pretty_print_index(dr.stop))"
end

"""
    Index(ix, child=Iden())

An indexing optic representing access to indices `ix`. A VarName{:x} with this optic
represents access to `x[ix...]`. The child optic represents any further indexing or
property access after this indexing operation.
"""
struct Index{I<:Tuple,C<:AbstractOptic} <: AbstractOptic
    ix::I
    child::C
end
Index(ix::Tuple, child::C=Iden()) where {C<:AbstractOptic} = Index{typeof(ix),C}(ix, child)

Base.:(==)(a::Index, b::Index) = a.ix == b.ix && a.child == b.child
Base.isequal(a::Index, b::Index) = a == b
function _pretty_print_optic(io::IO, idx::Index)
    ixs = join(map(_pretty_print_index, idx.ix), ", ")
    print(io, "[$(ixs)]")
    return _pretty_print_optic(io, idx.child)
end
function to_accessors(idx::Index)
    ilens = Accessors.IndexLens(idx.ix)
    return if idx.child isa Iden
        ilens
    else
        Base.ComposedFunction(to_accessors(idx.child), ilens)
    end
end
is_dynamic(idx::Index) = any(_is_dynamic_idx, idx.ix) || is_dynamic(idx.child)
function concretize(idx::Index, val)
    concretized_indices = if length(idx.ix) == 0
        []
    elseif length(idx.ix) == 1
        # If there's only one index, it's linear indexing. This code is mostly lifted from
        # Accessors.jl.
        [concretize(only(idx.ix), val, nothing)]
    else
        # If there are multiple indices, then each index corresponds to a different
        # dimension.
        [concretize(ix, val, dim) for (dim, ix) in enumerate(idx.ix)]
    end
    inner_concretized = concretize(idx.child, val[concretized_indices...])
    return Index((concretized_indices...,), inner_concretized)
end

"""
    Property{sym}(child=Iden())

A property access optic representing access to property `sym`. A VarName{:x} with this
optic represents access to `x.sym`. The child optic represents any further indexing
or property access after this property access operation.
"""
struct Property{sym,C<:AbstractOptic} <: AbstractOptic
    child::C
end
Property{sym}(child::C=Iden()) where {sym,C<:AbstractOptic} = Property{sym,C}(child)

Base.:(==)(a::Property{sym}, b::Property{sym}) where {sym} = a.child == b.child
Base.:(==)(a::Property, b::Property) = false
Base.isequal(a::Property, b::Property) = a == b
function _pretty_print_optic(io::IO, prop::Property{sym}) where {sym}
    print(io, ".$(sym)")
    return _pretty_print_optic(io, prop.child)
end
function to_accessors(prop::Property{sym}) where {sym}
    plens = Accessors.PropertyLens{sym}()
    return if prop.child isa Iden
        plens
    else
        Base.ComposedFunction(to_accessors(prop.child), plens)
    end
end
is_dynamic(prop::Property) = is_dynamic(prop.child)
function concretize(prop::Property{sym}, val) where {sym}
    inner_concretized = concretize(prop.child, getproperty(val, sym))
    return Property{sym}(inner_concretized)
end

function Base.:(∘)(outer::AbstractOptic, inner::AbstractOptic)
    if outer isa Iden
        return inner
    elseif inner isa Iden
        return outer
    else
        # TODO...
        error("not implemented")
    end
end

"""
    _head(optic)

Get the innermost layer of an AbstractOptic. For all optics, we have that `_tail(optic) ∘
_head(optic) == optic`.
"""
_head(::Property{s}) where {s} = Property{s}(Iden())
_head(idx::Index) = Index((idx.ix...,), Iden())
_head(i::Iden) = i

"""
    _tail(optic)

Get everything but the innermost layer of an optic. For all  optics, we have that
`_tail(optic) ∘ _head(optic) == optic`.
```
"""
_tail(p::Property) = p.child
_tail(idx::Index) = idx.child
_tail(i::Iden) = i

"""
    _last(optic)

Get the outermost layer of an optic. For all  optics, we have that `_last(optic) ∘
_init(optic) == optic`.
"""
function _last(p::Property{s}) where {s}
    if p.child isa Iden
        return p
    else
        return _last(p.child)
    end
end
function _last(idx::Index)
    if idx.child isa Iden
        return idx
    else
        return _last(idx.child)
    end
end
_last(i::Iden) = i

"""
    _init(optic)

Get everything but the outermost layer of an optic. For all optics, we have that
`_last(optic) ∘ _init(optic) == optic`.
"""
function _init(p::Property{s}) where {s}
    return if p.child isa Iden
        Iden()
    else
        Property{s}(_init(p.child))
    end
end
function _init(idx::Index)
    return if idx.child isa Iden
        Iden()
    else
        Index(idx.ix, _init(idx.child))
    end
end
_init(i::Iden) = i
