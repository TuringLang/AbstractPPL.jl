using Accessors: Accessors
using MacroTools: MacroTools

"""
    AbstractOptic

An abstract type that represents the non-symbol part of a VarName, i.e., the section of the
variable that is of interest. For example, in `x.a[1][2]`, the `AbstractOptic` represents
the `.a[1][2]` part.
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
is_dynamic(::Iden) = false
concretize(i::Iden, ::Any) = i
(::Iden)(obj) = obj
Accessors.set(obj::Any, ::Iden, val) = Accessors.set(obj, identity, val)

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

# The stored `Expr`

The `expr` field stores the original expression and is used both for pretty-printing as well
as comparisons.

Note that because the stored function `f` is an anonymous function that is generated
dynamically, we should not include it in equality comparisons, as two functions that are
actually equivalent will not compare equal:

```julia
julia> (x -> x + 1) == (x -> x + 1)
false
```

But, thankfully, we can just compare the `expr` field to determine whether the
DynamicIndices were constructed from the same expression (which implies that their
functions are equivalent).

Note that these definitions also allow us some degree of resilience towards whitespace
changes, or parenthesisation, in the original expression. For example, `begin+1` and `(begin
+ 1)` will be treated as the same expression. However, it does not handle commutative
expressions; e.g., `begin + 1` and `1 + begin` will be treated as different expressions.
"""
struct DynamicIndex{E<:Union{Expr,Symbol},F}
    expr::E
    f::F
end
Base.:(==)(a::DynamicIndex, b::DynamicIndex) = a.expr == b.expr
Base.isequal(a::DynamicIndex, b::DynamicIndex) = isequal(a.expr, b.expr)
Base.hash(di::DynamicIndex, h::UInt) = hash(di.expr, h)

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
    if has_begin_or_end(expr)
        replaced_expr = MacroTools.postwalk(x -> _make_dynamicindex_expr(x, val, dim), expr)
        return :(DynamicIndex($(QuoteNode(expr)), $val -> $replaced_expr))
    else
        return esc(expr)
    end
end
function _make_dynamicindex_expr(symbol::Symbol, val_sym::Symbol, dim::Union{Nothing,Int})
    # NOTE(penelopeysm): We could just use `:end` instead of Symbol(:end), but the former
    # messes up syntax highlighting with Treesitter
    # https://github.com/tree-sitter/tree-sitter-julia/issues/104
    if symbol === Symbol(:begin)
        return if dim === nothing
            :(Base.firstindex($val_sym))
        else
            :(Base.Fix2(firstindex, $dim)($val_sym))
        end
    elseif symbol === Symbol(:end)
        return if dim === nothing
            :(Base.lastindex($val_sym))
        else
            :(Base.Fix2(lastindex, $dim)($val_sym))
        end
    else
        # Just a variable; but we need to escape it to allow interpolation.
        return esc(symbol)
    end
end
function _make_dynamicindex_expr(i::Any, ::Symbol, ::Union{Nothing,Int})
    # this handles things like integers, colons, etc.
    return i
end

has_begin_or_end(expr::Expr) = has_begin_or_end_inner(expr, false)
function has_begin_or_end_inner(x, found::Bool)
    return found ||
           x ∈ (:end, :begin, Expr(:end), Expr(:begin)) ||
           (x isa Expr && any(arg -> has_begin_or_end_inner(arg, found), x.args))
end

_pretty_string_index(ix) = string(ix)
_pretty_string_index(::Colon) = ":"
_pretty_string_index(x::Symbol) = repr(x)
_pretty_string_index(x::String) = repr(x)
_pretty_string_index(di::DynamicIndex) = "DynamicIndex($(di.expr))"

_concretize_index(idx::Any, ::Any) = idx
_concretize_index(idx::DynamicIndex, val) = idx.f(val)

"""
    Index(ix, kw, child=Iden())

An indexing optic representing access to indices `ix`, which may also take the form of
keyword arguments `kw`. A VarName{:x} with this optic represents access to `x[ix...,
kw...]`. The child optic represents any further indexing or property access after this
indexing operation.
"""
struct Index{I<:Tuple,N<:NamedTuple,C<:AbstractOptic} <: AbstractOptic
    ix::I
    kw::N
    child::C
    function Index(ix::Tuple, kw::NamedTuple, child::C=Iden()) where {C<:AbstractOptic}
        return new{typeof(ix),typeof(kw),C}(ix, kw, child)
    end
end

Base.:(==)(a::Index, b::Index) = a.ix == b.ix && a.kw == b.kw && a.child == b.child
function Base.isequal(a::Index, b::Index)
    return isequal(a.ix, b.ix) && isequal(a.kw, b.kw) && isequal(a.child, b.child)
end
Base.hash(a::Index, h::UInt) = hash((a.ix, a.kw, a.child), h)
function _pretty_print_optic(io::IO, idx::Index)
    ixs = collect(map(_pretty_string_index, idx.ix))
    kws = map(
        kv -> "$(kv.first)=$(_pretty_string_index(kv.second))", collect(pairs(idx.kw))
    )
    print(io, "[$(join(vcat(ixs, kws), ", "))]")
    return _pretty_print_optic(io, idx.child)
end
is_dynamic(idx::Index) = any(ix -> ix isa DynamicIndex, idx.ix) || is_dynamic(idx.child)

# Helper function to decide whether to use `view` or `getindex`. For AbstractArray, the
# default behaviour is to attempt to use a view.
_maybe_view(val::AbstractArray, i...; k...) = view(val, i...; k...)
# If it's just a single element, don't use a view, as that returns a weird 0-dimensional
# SubArray (rather than an element) that messes things up if there are further layers of
# optics. For example, if it's an Array of NamedTuples, then trying to access fields on that
# 0-dimensional SubArray will fail.
_maybe_view(val::AbstractArray, i::Int...) = getindex(val, i...)
# Other things like dictionaries can't be `view`ed into.
_maybe_view(val, i...; k...) = getindex(val, i...; k...)

function concretize(idx::Index, val)
    concretized_indices = tuple(map(Base.Fix2(_concretize_index, val), idx.ix)...)
    inner_concretized = if idx.child isa Iden
        # Explicitly having this branch allows us to shortcircuit _maybe_view(...), which
        # can error if val[concretized_indices...] is an UndefInitializer. Note that if
        # val[concretized_indices...] is an UndefInitializer, then it is not meaningful for
        # `idx.child` to be anything other than `Iden` anyway, since there is nothing to
        # further index into.
        Iden()
    else
        concretize(idx.child, _maybe_view(val, concretized_indices...; idx.kw...))
    end
    return Index(concretized_indices, idx.kw, inner_concretized)
end
function (idx::Index)(obj)
    cidx = concretize(idx, obj)
    return cidx.child(Base.getindex(obj, cidx.ix...; cidx.kw...))
end
function Accessors.set(obj, idx::Index, newval)
    cidx = concretize(idx, obj)
    inner_newval = if idx.child isa Iden
        newval
    else
        inner_obj = Base.getindex(obj, cidx.ix...; cidx.kw...)
        Accessors.set(inner_obj, idx.child, newval)
    end
    return if !isempty(cidx.kw)
        # `Accessors.IndexLens` does not handle keyword arguments so we need to do this
        # ourselves. Note that the following code essentially assumes that `obj` is an
        # AbstractArray or similar type that directly implements `setindex!`.
        newobj = similar(obj)
        copy!(newobj, obj)
        Base.setindex!(newobj, inner_newval, cidx.ix...; cidx.kw...)
        newobj
    else
        # Defer to Accessors' implementation, so that we don't have to reinvent the wheel
        # (well, not more than what we have already done...). This is helpful because
        # Accessors implements a lot of methods for different types of `obj`.
        Accessors.set(obj, Accessors.IndexLens(cidx.ix), inner_newval)
    end
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
is_dynamic(prop::Property) = is_dynamic(prop.child)
function concretize(prop::Property{sym}, val) where {sym}
    inner_concretized = concretize(prop.child, getproperty(val, sym))
    return Property{sym}(inner_concretized)
end
function (prop::Property{sym})(obj) where {sym}
    return prop.child(getproperty(obj, sym))
end
function Accessors.set(obj, prop::Property{sym}, newval) where {sym}
    inner_obj = getproperty(obj, sym)
    inner_newval = Accessors.set(inner_obj, prop.child, newval)
    # Defer to Accessors' implementation again.
    return Accessors.set(obj, Accessors.PropertyLens{sym}(), inner_newval)
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
            return Index(inner.ix, inner.kw, outer ∘ inner.child)
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
ohead(idx::Index) = Index(idx.ix, idx.kw, Iden())
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
        Index(idx.ix, idx.kw, oinit(idx.child))
    end
end
oinit(i::Iden) = i
