using Accessors: Accessors

"""
    AbstractOptic

An abstract type that represents the non-symbol part of a VarName, i.e., the section of the
variable that is of interest. For example, in `x.a[1][2]`, the `AbstractOptic` represents
the `.a[1][2]` part.

# Interface

This is WIP.

- Base.show
- to_accessors(optic) -> Accessors.Lens (recovering the old representation)

Not sure if we want to introduce getters and setters and BangBang-style stuff.
"""
abstract type AbstractOptic end
function Base.show(io::IO, optic::AbstractOptic)
    print(io, "Optic(")
    pretty_print_optic(io, optic)
    return print(io, ")")
end

"""
    Iden()

The identity optic. This is the optic used when we are referring to the entire variable.
It is also the base case for composing optics.
"""
struct Iden <: AbstractOptic end
pretty_print_optic(::IO, ::Iden) = nothing
to_accessors(::Iden) = identity
concretize(i::Iden, ::Any) = i

"""
    DynamicIndex

An abstract type representing dynamic indices such as `begin`, `end`, and `:`. These indices
are things which cannot be resolved until we provide the value that is being indexed into.
When parsing VarNames, we convert such indices into subtypes of `DynamicIndex`, and we later
mark them as requiring concretisation.
"""
abstract type DynamicIndex end
# Fallback for all other indices
concretize(@nospecialize(ix::Any), ::Any, ::Any) = ix

struct DynamicBegin <: DynamicIndex end
concretize(::DynamicBegin, val, dim::Nothing) = Base.firstindex(val)
concretize(::DynamicBegin, val, dim) = Base.firstindex(val, dim)

struct DynamicEnd <: DynamicIndex end
concretize(::DynamicEnd, val, dim::Nothing) = Base.lastindex(val)
concretize(::DynamicEnd, val, dim) = Base.lastindex(val, dim)

struct DynamicColon <: DynamicIndex end
concretize(::DynamicColon, val, dim::Nothing) = Base.firstindex(val):Base.lastindex(val)
concretize(::DynamicColon, val, dim) = Base.firstindex(val, dim):Base.lastindex(val, dim)

struct DynamicRange{T1,T2} <: DynamicIndex
    start::T1
    stop::T2
end
function concretize(dr::DynamicRange, axis)
    start = dr.start isa DynamicIndex ? concretize(dr.start, axis) : dr.start
    stop = dr.stop isa DynamicIndex ? concretize(dr.stop, axis) : dr.stop
    return start:stop
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
function pretty_print_optic(io::IO, idx::Index)
    ixs = join(idx.ix, ", ")
    print(io, "[$(ixs)]")
    return pretty_print_optic(io, idx.child)
end
function to_accessors(idx::Index)
    ilens = Accessors.IndexLens(idx.ix)
    return if idx.child isa Iden
        ilens
    else
        Base.ComposedFunction(to_accessors(idx.child), ilens)
    end
end
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
function pretty_print_optic(io::IO, prop::Property{sym}) where {sym}
    print(io, ".$(sym)")
    return pretty_print_optic(io, prop.child)
end
function to_accessors(prop::Property{sym}) where {sym}
    plens = Accessors.PropertyLens{sym}()
    return if prop.child isa Iden
        plens
    else
        Base.ComposedFunction(to_accessors(prop.child), plens)
    end
end
function concretize(prop::Property{sym}, val) where {sym}
    inner_concretized = concretize(prop.child, getproperty(val, sym))
    return Property{sym}(inner_concretized)
end
