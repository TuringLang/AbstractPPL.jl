using Accessors: Accessors
using MacroTools: MacroTools

"""
    AbstractOptic

An abstract type that represents the non-symbol part of a VarName, i.e., the section of the
variable that is of interest. For example, in `x.a[1][2]`, the `AbstractOptic` represents
the `.a[1][2]` part.

# Public interface

TODO

- Base.show
- Base.:(==), Base.isequal
- Base.:(∘) (composition)
- ohead, otail, olast, oinit (decomposition)

- to_accessors(optic) -> Accessors.Lens (recovering the old representation)
- is_dynamic(optic) -> Bool (whether the optic contains any dynamic indices)
- concretize(optic, val) -> AbstractOptic (resolving any dynamic indices given the value)

We probably want to introduce getters and setters. See e.g.
https://juliaobjects.github.io/Accessors.jl/stable/examples/custom_macros/
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
When parsing VarNames, we convert such indices into subtypes of `DynamicIndex`.

Because a `DynamicIndex` cannot be resolved until we have the value being indexed into, it
is actually a wrapper around a function that, when called on the value, returns the concrete
index.

For example:

- the index `begin` is turned into `DynamicIndex(:begin, (val) -> Base.firstindex(val))`.
- the index `1:end` is turned into `DynamicIndex(:(1:end), (val) -> 1:Base.lastindex(val))`.

The `expr` field stores the original expression solely for pretty-printing purposes.
"""
struct DynamicIndex{E<:Union{Expr,Symbol},F}
    expr::E
    f::F
end
function _make_dynamicindex_expr(symbol::Symbol, dim::Union{Nothing,Int})
    # NOTE(penelopeysm): We could just use `:end` instead of Symbol(:end), but the former
    # messes up syntax highlighting with Treesitter
    # https://github.com/tree-sitter/tree-sitter-julia/issues/104
    if symbol === Symbol(:begin)
        func = dim === nothing ? :(Base.firstindex) : :(Base.Fix2(firstindex, $dim))
        return :(DynamicIndex($(QuoteNode(symbol)), $func))
    elseif symbol === Symbol(:end)
        func = dim === nothing ? :(Base.lastindex) : :(Base.Fix2(lastindex, $dim))
        return :(DynamicIndex($(QuoteNode(symbol)), $func))
    else
        # Just a variable; but we need to escape it to allow interpolation.
        return esc(symbol)
    end
end
function _make_dynamicindex_expr(expr::Expr, dim::Union{Nothing,Int})
    @gensym val
    replaced_expr = MacroTools.postwalk(x -> replace_begin_and_end(x, val, dim), expr)
    return if replaced_expr == expr
        # Nothing to replace, just use the original expr.
        expr
    else
        :(DynamicIndex($(QuoteNode(expr)), $val -> $replaced_expr))
    end
end

# Replace all instances of `begin` in `expr` with `_firstindex_dim(val, dim)` and
# all instances of `end` with `_lastindex_dim(val, dim)`.
replace_begin_and_end(x, ::Any, ::Any) = x
function replace_begin_and_end(x::Symbol, val_sym, dim)
    return if (x === :begin)
        dim === nothing ? :(Base.firstindex($val_sym)) : :(Base.firstindex($val_sym, $dim))
    elseif (x === :end)
        dim === nothing ? :(Base.lastindex($val_sym)) : :(Base.lastindex($val_sym, $dim))
    else
        # It's some other symbol; we need to escape it to allow interpolation.
        esc(x)
    end
end
_pretty_string_index(ix) = string(ix)
_pretty_string_index(::Colon) = ":"
_pretty_string_index(di::DynamicIndex) = "DynamicIndex($(di.expr))"

_concretize_index(idx::Any, ::Any) = idx
_concretize_index(idx::DynamicIndex, val) = idx.f(val)

"""
    Index(ix, child=Iden())

An indexing optic representing access to indices `ix`. A VarName{:x} with this optic
represents access to `x[ix...]`. The child optic represents any further indexing or
property access after this indexing operation.
"""
struct Index{I<:Tuple,C<:AbstractOptic} <: AbstractOptic
    ix::I
    child::C
    function Index(ix::Tuple, child::C=Iden()) where {C<:AbstractOptic}
        return new{typeof(ix),C}(ix, child)
    end
end

Base.:(==)(a::Index, b::Index) = a.ix == b.ix && a.child == b.child
Base.isequal(a::Index, b::Index) = a == b
function _pretty_print_optic(io::IO, idx::Index)
    ixs = join(map(_pretty_string_index, idx.ix), ", ")
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
is_dynamic(idx::Index) = any(ix -> ix isa DynamicIndex, idx.ix) || is_dynamic(idx.child)
function concretize(idx::Index, val)
    concretized_indices = map(Base.Fix2(_concretize_index, val), idx.ix)
    inner_concretized = concretize(idx.child, view(val, concretized_indices...))
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
getsym(::Property{s}) where {s} = s
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

"""
    ∘(outer::AbstractOptic, inner::AbstractOptic)

Compose two `AbstractOptic`s together.

```jldoctest
julia> p1 = @opticof(_.a[1])
Optic(.a[1])

julia> p2 = @opticof(_.b[2, 3])
Optic(.b[2, 3])

julia> p1 ∘ p2
Optic(.b[2, 3].a[1])
```
"""
function Base.:(∘)(outer::AbstractOptic, inner::AbstractOptic)
    if outer isa Iden
        return inner
    elseif inner isa Iden
        return outer
    else
        if inner isa Property
            return Property{getsym(inner)}(outer ∘ inner.child)
        elseif inner isa Index
            return Index(inner.ix, outer ∘ inner.child)
        else
            error("unreachable; unknown AbstractOptic subtype $(typeof(inner))")
        end
    end
end

"""
    cat(optics::AbstractOptic...)

Compose multiple `AbstractOptic`s together. The optics should be provided from
innermost to outermost, i.e., `cat(o1, o2, o3)` corresponds to `o3 ∘ o2 ∘ o1`.

"""
function Base.cat(optics::AbstractOptic...)
    return foldl((a, b) -> b ∘ a, optics; init=Iden())
end

"""
    ohead(optic::AbstractOptic)

Get the innermost layer of an optic. For all optics, we have that `otail(optic) ∘
ohead(optic) == optic`.

```jldoctest
julia> ohead(@opticof _.a[1][2])
Optic(.a)

julia> ohead(@opticof _)
Optic()
```
"""
ohead(::Property{s}) where {s} = Property{s}(Iden())
ohead(idx::Index) = Index((idx.ix...,), Iden())
ohead(i::Iden) = i

"""
    otail(optic::AbstractOptic)

Get everything but the innermost layer of an optic. For all optics, we have that
`otail(optic) ∘ ohead(optic) == optic`.

```jldoctest
julia> otail(@opticof _.a[1][2])
Optic([1][2])

julia> otail(@opticof _)
Optic()
```
"""
otail(p::Property) = p.child
otail(idx::Index) = idx.child
otail(i::Iden) = i

"""
    olast(optic::AbstractOptic)

Get the outermost layer of an optic. For all optics, we have that `olast(optic) ∘
oinit(optic) == optic`.

```jldoctest
julia> olast(@opticof _.a[1][2])
Optic([2])

julia> olast(@opticof _)
Optic()
```
"""
function olast(p::Property{s}) where {s}
    if p.child isa Iden
        return p
    else
        return olast(p.child)
    end
end
function olast(idx::Index)
    if idx.child isa Iden
        return idx
    else
        return olast(idx.child)
    end
end
olast(i::Iden) = i

"""
    oinit(optic::AbstractOptic)

Get everything but the outermost layer of an optic. For all optics, we have that
`olast(optic) ∘ oinit(optic) == optic`.

```jldoctest
julia> oinit(@opticof _.a[1][2])
Optic(.a[1])

julia> oinit(@opticof _)
Optic()
```
"""
function oinit(p::Property{s}) where {s}
    return if p.child isa Iden
        Iden()
    else
        Property{s}(oinit(p.child))
    end
end
function oinit(idx::Index)
    return if idx.child isa Iden
        Iden()
    else
        Index(idx.ix, oinit(idx.child))
    end
end
oinit(i::Iden) = i
