using Accessors
using Accessors: PropertyLens, IndexLens, DynamicIndexLens

# nb. ComposedFunction is the same as Accessors.ComposedOptic
const ALLOWED_OPTICS = Union{typeof(identity),PropertyLens,IndexLens,ComposedFunction}

"""
    VarName{sym}(optic=identity)

A variable identifier for a symbol `sym` and optic `optic`.

The Julia variable in the model corresponding to `sym` can refer to a single value or to a
hierarchical array structure of univariate, multivariate or matrix variables. The field `lens`
stores the indices requires to access the random variable from the Julia variable indicated by `sym`
as a tuple of tuples. Each element of the tuple thereby contains the indices of one optic
operation.

`VarName`s can be manually constructed using the `VarName{sym}(optic)` constructor, or from an
optic expression through the [`@varname`](@ref) convenience macro.

# Examples

```jldoctest; setup=:(using Accessors)
julia> vn = VarName{:x}(Accessors.IndexLens((Colon(), 1)) ⨟ Accessors.IndexLens((2, )))
x[:, 1][2]

julia> getoptic(vn)
(@o _[Colon(), 1][2])

julia> @varname x[:, 1][1+1]
x[:, 1][2]
```
"""
struct VarName{sym,T<:ALLOWED_OPTICS}
    optic::T

    function VarName{sym}(optic=identity) where {sym}
        optic = normalise(optic)
        if !is_static_optic(typeof(optic))
            throw(
                ArgumentError(
                    "attempted to construct `VarName` with unsupported optic of type $(nameof(typeof(optic)))",
                ),
            )
        end
        return new{sym,typeof(optic)}(optic)
    end
end

"""
    is_static_optic(l)

Return `true` if `l` is one or a composition of `identity`, `PropertyLens`, and `IndexLens`; `false` if `l` is 
one or a composition of `DynamicIndexLens`; and undefined otherwise.
"""
is_static_optic(::Type{<:Union{typeof(identity),PropertyLens,IndexLens}}) = true
function is_static_optic(::Type{ComposedFunction{LO,LI}}) where {LO,LI}
    return is_static_optic(LO) && is_static_optic(LI)
end
is_static_optic(::Type{<:DynamicIndexLens}) = false

"""
    normalise(optic)

Enforce that compositions of optics are always nested in the same way, in that
a ComposedFunction never has a ComposedFunction as its inner lens. Thus, for
example,

```jldoctest; setup=:(using Accessors)
julia> op1 = ((@o _.c) ∘ (@o _.b)) ∘ (@o _.a)
(@o _.a.b.c)

julia> op2 = (@o _.c) ∘ ((@o _.b) ∘ (@o _.a))
(@o _.c) ∘ ((@o _.a.b))

julia> op1 == op2
false

julia> AbstractPPL.normalise(op1) == AbstractPPL.normalise(op2) == @o _.a.b.c
true
```

This function also removes redundant `identity` optics from ComposedFunctions:

```jldoctest; setup=:(using Accessors)
julia> op3 = ((@o _.b) ∘ identity) ∘ (@o _.a)
(@o identity(_.a).b)

julia> op4 = (@o _.b) ∘ (identity ∘ (@o _.a))
(@o _.b) ∘ ((@o identity(_.a)))

julia> AbstractPPL.normalise(op3) == AbstractPPL.normalise(op4) == @o _.a.b
true
```
"""
function normalise(o::ComposedFunction{Outer,<:ComposedFunction}) where {Outer}
    # `o` is currently (outer ∘ (inner_outer ∘ inner_inner)).
    # We want to change this to:
    # o = (outer ∘ inner_outer) ∘ inner_inner
    inner_inner = o.inner.inner
    inner_outer = o.inner.outer
    # Recursively call normalise because inner_inner could itself be a
    # ComposedFunction
    return normalise((o.outer ∘ inner_outer) ∘ inner_inner)
end
function normalise(o::ComposedFunction{Outer,typeof(identity)} where {Outer})
    # strip outer identity
    return normalise(o.outer)
end
function normalise(o::ComposedFunction{typeof(identity),Inner} where {Inner})
    # strip inner identity
    return normalise(o.inner)
end
normalise(o::ComposedFunction) = normalise(o.outer) ∘ o.inner
normalise(o::ALLOWED_OPTICS) = o
# These two methods are needed to avoid method ambiguity.
normalise(o::ComposedFunction{typeof(identity),<:ComposedFunction}) = normalise(o.inner)
normalise(::ComposedFunction{typeof(identity),typeof(identity)}) = identity

"""
    getsym(vn::VarName)

Return the symbol of the Julia variable used to generate `vn`.

## Examples

```jldoctest
julia> getsym(@varname(x[1][2:3]))
:x

julia> getsym(@varname(y))
:y
```
"""
getsym(::VarName{sym}) where {sym} = sym

"""
    getoptic(vn::VarName)

Return the optic of the Julia variable used to generate `vn`.

## Examples

```jldoctest
julia> getoptic(@varname(x[1][2:3]))
(@o _[1][2:3])

julia> getoptic(@varname(y))
identity (generic function with 1 method)
```
"""
getoptic(vn::VarName) = vn.optic

"""
    get(obj, vn::VarName{sym})

Alias for `(PropertyLens{sym}() ⨟ getoptic(vn))(obj)`.
```
"""
function Base.get(obj, vn::VarName{sym}) where {sym}
    return (PropertyLens{sym}() ⨟ getoptic(vn))(obj)
end

"""
    set(obj, vn::VarName{sym}, value)

Alias for `set(obj, PropertyLens{sym}() ⨟ getoptic(vn), value)`.

# Example

```jldoctest; setup = :(using AbstractPPL: Accessors; nt = (a = 1, b = (c = [1, 2, 3],)); name = :nt)
julia> Accessors.set(nt, @varname(a), 10)
(a = 10, b = (c = [1, 2, 3],))

julia> Accessors.set(nt, @varname(b.c[1]), 10)
(a = 1, b = (c = [10, 2, 3],))
```
"""
function Accessors.set(obj, vn::VarName{sym}, value) where {sym}
    return Accessors.set(obj, PropertyLens{sym}() ⨟ getoptic(vn), value)
end

# Allow compositions with optic.
function Base.:∘(optic::ALLOWED_OPTICS, vn::VarName{sym}) where {sym}
    return VarName{sym}(optic ∘ getoptic(vn))
end

Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getoptic(vn)), h)
function Base.:(==)(x::VarName, y::VarName)
    return getsym(x) == getsym(y) && getoptic(x) == getoptic(y)
end

function Base.show(io::IO, vn::VarName{sym,T}) where {sym,T}
    print(io, getsym(vn))
    return _show_optic(io, getoptic(vn))
end

# modified from https://github.com/JuliaObjects/Accessors.jl/blob/01528a81fdf17c07436e1f3d99119d3f635e4c26/src/sugar.jl#L502
function _show_optic(io::IO, optic)
    opts = Accessors.deopcompose(optic)
    inner = Iterators.takewhile(x -> applicable(_shortstring, "", x), opts)
    outer = Iterators.dropwhile(x -> applicable(_shortstring, "", x), opts)
    if !isempty(outer)
        show(io, opcompose(outer...))
        print(io, " ∘ ")
    end
    shortstr = reduce(_shortstring, inner; init="")
    return print(io, shortstr)
end

_shortstring(prev, o::IndexLens) = "$prev[$(join(map(prettify_index, o.indices), ", "))]"
_shortstring(prev, ::typeof(identity)) = "$prev"
_shortstring(prev, o) = Accessors._shortstring(prev, o)

prettify_index(x) = repr(x)
prettify_index(::Colon) = ":"

"""
    Symbol(vn::VarName)

Return a `Symbol` representation of the variable identifier `VarName`.

# Examples
```jldoctest
julia> Symbol(@varname(x[1][2:3]))
Symbol("x[1][2:3]")

julia> Symbol(@varname(x[1][:]))
Symbol("x[1][:]")
```
"""
Base.Symbol(vn::VarName) = Symbol(string(vn))  # simplified symbol

"""
    ConcretizedSlice(::Base.Slice)

An indexing object wrapping the range of a `Base.Slice` object representing the concrete indices a
`:` indicates.  Behaves the same, but prints differently, namely, still as `:`.
"""
struct ConcretizedSlice{T,R} <: AbstractVector{T}
    range::R
end

function ConcretizedSlice(s::Base.Slice{R}) where {R}
    return ConcretizedSlice{eltype(s.indices),R}(s.indices)
end
Base.show(io::IO, s::ConcretizedSlice) = print(io, ":")
function Base.show(io::IO, ::MIME"text/plain", s::ConcretizedSlice)
    return print(io, "ConcretizedSlice(", s.range, ")")
end
Base.size(s::ConcretizedSlice) = size(s.range)
Base.iterate(s::ConcretizedSlice, state...) = Base.iterate(s.range, state...)
Base.collect(s::ConcretizedSlice) = collect(s.range)
Base.getindex(s::ConcretizedSlice, i) = s.range[i]
Base.hasfastin(::Type{<:ConcretizedSlice}) = true
Base.in(i, s::ConcretizedSlice) = i in s.range

# and this is the reason why we are doing this:
Base.to_index(A, s::ConcretizedSlice) = Base.Slice(s.range)

"""
    reconcretize_index(original_index, lowered_index)

Create the index to be emitted in `concretize`.  `original_index` is the original, unconcretized
index, and `lowered_index` the respective position of the result of `to_indices`.

The only purpose of this are special cases like `:`, which we want to avoid becoming a
`Base.Slice(OneTo(...))` -- it would confuse people when printed.  Instead, we concretize to a
`ConcretizedSlice` based on the `lowered_index`, just what you'd get with an explicit `begin:end`
"""
reconcretize_index(original_index, lowered_index) = lowered_index
function reconcretize_index(original_index::Colon, lowered_index::Base.Slice)
    return ConcretizedSlice(lowered_index)
end

"""
    concretize(l, x)

Return `l` instantiated on `x`, i.e. any information related to the runtime shape of `x` is
evaluated. This concerns `begin`, `end`, and `:` slices.

Basically, every index is converted to a concrete value using `Base.to_index` on `x`.  However, `:`
slices are only converted to `ConcretizedSlice` (as opposed to `Base.Slice{Base.OneTo}`), to keep
the result close to the original indexing.
"""
concretize(I::ALLOWED_OPTICS, x) = I
concretize(I::DynamicIndexLens, x) = concretize(IndexLens(I.f(x)), x)
function concretize(I::IndexLens, x)
    return IndexLens(reconcretize_index.(I.indices, to_indices(x, I.indices)))
end
function concretize(I::ComposedFunction, x)
    x_inner = I.inner(x) # TODO: get view here
    return ComposedFunction(concretize(I.outer, x_inner), concretize(I.inner, x))
end

"""
    concretize(vn::VarName, x)

Return `vn` concretized on `x`, i.e. any information related to the runtime shape of `x` is
evaluated. This concerns `begin`, `end`, and `:` slices.

# Examples
```jldoctest; setup=:(using Accessors)
julia> x = (a = [1.0 2.0; 3.0 4.0; 5.0 6.0], );

julia> getoptic(@varname(x.a[1:end, end][:], true)) # concrete=true required for @varname
(@o _.a[1:3, 2][:])

julia> y = zeros(10, 10);

julia> @varname(y[:], true)
y[:]

julia> # The underlying value is concretized, though:
       AbstractPPL.getoptic(AbstractPPL.concretize(@varname(y[:]), y)).indices[1]
ConcretizedSlice(Base.OneTo(100))
```
"""
concretize(vn::VarName{sym}, x) where {sym} = VarName{sym}(concretize(getoptic(vn), x))

"""
    @varname(expr, concretize=false)

A macro that returns an instance of [`VarName`](@ref) given a symbol or indexing expression `expr`.

If `concretize` is `true`, the resulting expression will be wrapped in a `concretize()` call.

Note that expressions involving dynamic indexing, i.e. `begin` and/or `end`, will always need to be
concretized as `VarName` only supports non-dynamic indexing as determined by
`is_static_optic`. See examples below.

## Examples

### Dynamic indexing
```jldoctest
julia> x = (a = [1.0 2.0; 3.0 4.0; 5.0 6.0], );

julia> @varname(x.a[1:end, end][:], true)
x.a[1:3, 2][:]

julia> @varname(x.a[end], false)  # disable concretization
ERROR: LoadError: Variable name `x.a[end]` is dynamic and requires concretization!
[...]

julia> @varname(x.a[end])  # concretization occurs by default if deemed necessary
x.a[6]

julia> # Note that "dynamic" here refers to usage of `begin` and/or `end`,
       # _not_ "information only available at runtime", i.e. the following works.
       [@varname(x.a[i]) for i = 1:length(x.a)][end]
x.a[6]

julia> # Potentially surprising behaviour, but this is equivalent to what Base does:
       @varname(x[2:2:5]), 2:2:5
(x[2:2:4], 2:2:4)
```

### General indexing

Under the hood `optic`s are used for the indexing:

```jldoctest
julia> getoptic(@varname(x))
identity (generic function with 1 method)

julia> getoptic(@varname(x[1]))
(@o _[1])

julia> getoptic(@varname(x[:, 1]))
(@o _[Colon(), 1])

julia> getoptic(@varname(x[:, 1][2]))
(@o _[Colon(), 1][2])

julia> getoptic(@varname(x[1,2][1+5][45][3]))
(@o _[1, 2][6][45][3])
```

This also means that we support property access:

```jldoctest
julia> getoptic(@varname(x.a))
(@o _.a)

julia> getoptic(@varname(x.a[1]))
(@o _.a[1])

julia> x = (a = [(b = rand(2), )], ); getoptic(@varname(x.a[1].b[end], true))
(@o _.a[1].b[2])
```

Interpolation can be used for variable names, or array name, but not the lhs of a `.` expression.
Variables within indices are always evaluated in the calling scope.

```jldoctest
julia> name, i = :a, 10;

julia> @varname(x.\$name[i, i+1])
x.a[10, 11]

julia> @varname(\$name)
a

julia> @varname(\$name[1])
a[1]

julia> @varname(\$name.x[1])
a.x[1]

julia> @varname(b.\$name.x[1])
b.a.x[1]
```
"""
macro varname(expr::Union{Expr,Symbol}, concretize::Bool=Accessors.need_dynamic_optic(expr))
    return varname(expr, concretize)
end

varname(sym::Symbol) = :($(AbstractPPL.VarName){$(QuoteNode(sym))}())
varname(sym::Symbol, _) = varname(sym)
function varname(expr::Expr, concretize=Accessors.need_dynamic_optic(expr))
    if Meta.isexpr(expr, :ref) || Meta.isexpr(expr, :.)
        # Split into object/base symbol and lens.
        sym_escaped, optics = _parse_obj_optic(expr)
        # Setfield.jl escapes the return symbol, so we need to unescape
        # to call `QuoteNode` on it.
        sym = drop_escape(sym_escaped)

        # This is to handle interpolated heads -- Setfield treats them differently:
        # julia>  AbstractPPL._parse_obj_optics(Meta.parse("\$name.a"))
        # (:($(Expr(:escape, :_))), (:($(Expr(:escape, :name))), :((PropertyLens){:a}())))
        # julia> AbstractPPL._parse_obj_optic(:(x.a))
        # (:($(Expr(:escape, :x))), :(Accessors.opticcompose((PropertyLens){:a}())))
        if sym != :_
            sym = QuoteNode(sym)
        else
            sym = optics.args[2]
            optics = Expr(:call, optics.args[1], optics.args[3:end]...)
        end

        if concretize
            return :($(AbstractPPL.VarName){$sym}(
                $(AbstractPPL.concretize)($optics, $sym_escaped)
            ))
        elseif Accessors.need_dynamic_optic(expr)
            error("Variable name `$(expr)` is dynamic and requires concretization!")
        else
            return :($(AbstractPPL.VarName){$sym}($optics))
        end
    elseif Meta.isexpr(expr, :$, 1)
        return :($(AbstractPPL.VarName){$(esc(expr.args[1]))}())
    else
        error("Malformed variable name `$(expr)`!")
    end
end

drop_escape(x) = x
function drop_escape(expr::Expr)
    Meta.isexpr(expr, :escape) && return drop_escape(expr.args[1])
    return Expr(expr.head, map(x -> drop_escape(x), expr.args)...)
end

function _parse_obj_optic(ex)
    obj, optics = _parse_obj_optics(ex)
    optic = Expr(:call, Accessors.opticcompose, optics...)
    return obj, optic
end

# Accessors doesn't have the same support for interpolation
# so this function is copied and altered from `Setfield._parse_obj_lens`
function _parse_obj_optics(ex)
    if Meta.isexpr(ex, :$, 1)
        return esc(:_), (esc(ex.args[1]),)
    elseif Meta.isexpr(ex, :ref) && !isempty(ex.args)
        front, indices... = ex.args
        obj, frontoptics = _parse_obj_optics(front)
        if any(Accessors.need_dynamic_optic, indices)
            @gensym collection
            indices = Accessors.replace_underscore.(indices, collection)
            dims = length(indices) == 1 ? nothing : 1:length(indices)
            lindices = esc.(Accessors.lower_index.(collection, indices, dims))
            optics =
                :($(Accessors.DynamicIndexLens)($(esc(collection)) -> ($(lindices...),)))
        else
            index = esc(Expr(:tuple, indices...))
            optics = :($(Accessors.IndexLens)($index))
        end
    elseif Meta.isexpr(ex, :., 2)
        front = ex.args[1]
        property = ex.args[2].value # ex.args[2] is a QuoteNode
        obj, frontoptics = _parse_obj_optics(front)
        if property isa Union{Symbol,String}
            optics = :($(Accessors.PropertyLens){$(QuoteNode(property))}())
        elseif Meta.isexpr(property, :$, 1)
            optics = :($(Accessors.PropertyLens){$(esc(property.args[1]))}())
        else
            throw(
                ArgumentError(
                    string(
                        "Error while parsing :($ex). Second argument to `getproperty` can only be",
                        "a `Symbol` or `String` literal, received `$property` instead.",
                    ),
                ),
            )
        end
    else
        obj = esc(ex)
        return obj, ()
    end
    return obj, tuple(frontoptics..., optics)
end

"""
    @vsym(expr)

A macro that returns the variable symbol given the input variable expression `expr`.
For example, `@vsym x[1]` returns `:x`.

## Examples

```jldoctest
julia> @vsym x
:x

julia> @vsym x[1,1][2,3]
:x

julia> @vsym x[end]
:x
```
"""
macro vsym(expr::Union{Expr,Symbol})
    return QuoteNode(vsym(expr))
end

"""
    vsym(expr)

Return name part of the [`@varname`](@ref)-compatible expression `expr` as a symbol for input of the
[`VarName`](@ref) constructor.
"""
function vsym end

vsym(expr::Symbol) = expr
function vsym(expr::Expr)
    if Meta.isexpr(expr, :ref) || Meta.isexpr(expr, :.)
        return vsym(expr.args[1])
    else
        error("Malformed variable name `$(expr)`!")
    end
end

"""
    _head(optic)

Get the innermost layer of an optic.

For all (normalised) optics, we have that `normalise(_tail(optic) ∘
_head(optic) == optic)`.

!!! note
    Does not perform optic normalisation on the input. You may wish to call
    `normalise(optic)` before using this function if the optic you are passing
    was not obtained from a VarName.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._head(Accessors.@o _.a.b.c)
(@o _.a)

julia> AbstractPPL._head(Accessors.@o _[1][2][3])
(@o _[1])

julia> AbstractPPL._head(Accessors.@o _.a)
(@o _.a)

julia> AbstractPPL._head(Accessors.@o _[1])
(@o _[1])

julia> AbstractPPL._head(Accessors.@o _)
identity (generic function with 1 method)
```
"""
_head(o::ComposedFunction{Outer,Inner}) where {Outer,Inner} = o.inner
_head(o::Accessors.PropertyLens) = o
_head(o::Accessors.IndexLens) = o
_head(::typeof(identity)) = identity

"""
    _tail(optic)

Get everything but the innermost layer of an optic.

For all (normalised) optics, we have that `normalise(_tail(optic) ∘
_head(optic) == optic)`.

!!! note
    Does not perform optic normalisation on the input. You may wish to call
    `normalise(optic)` before using this function if the optic you are passing
    was not obtained from a VarName.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._tail(Accessors.@o _.a.b.c)
(@o _.b.c)

julia> AbstractPPL._tail(Accessors.@o _[1][2][3])
(@o _[2][3])

julia> AbstractPPL._tail(Accessors.@o _.a)
identity (generic function with 1 method)

julia> AbstractPPL._tail(Accessors.@o _[1])
identity (generic function with 1 method)

julia> AbstractPPL._tail(Accessors.@o _)
identity (generic function with 1 method)
```
"""
_tail(o::ComposedFunction{Outer,Inner}) where {Outer,Inner} = o.outer
_tail(::Accessors.PropertyLens) = identity
_tail(::Accessors.IndexLens) = identity
_tail(::typeof(identity)) = identity

"""
    _last(optic)

Get the outermost layer of an optic.

For all (normalised) optics, we have that `normalise(_last(optic) ∘
_init(optic)) == optic`.

!!! note
    Does not perform optic normalisation on the input. You may wish to call
    `normalise(optic)` before using this function if the optic you are passing
    was not obtained from a VarName.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._last(Accessors.@o _.a.b.c)
(@o _.c)

julia> AbstractPPL._last(Accessors.@o _[1][2][3])
(@o _[3])

julia> AbstractPPL._last(Accessors.@o _.a)
(@o _.a)

julia> AbstractPPL._last(Accessors.@o _[1])
(@o _[1])

julia> AbstractPPL._last(Accessors.@o _)
identity (generic function with 1 method)
```
"""
_last(o::ComposedFunction{Outer,Inner}) where {Outer,Inner} = _last(o.outer)
_last(o::Accessors.PropertyLens) = o
_last(o::Accessors.IndexLens) = o
_last(::typeof(identity)) = identity

"""
    _init(optic)

Get everything but the outermost layer of an optic.

For all (normalised) optics, we have that `normalise(_last(optic) ∘
_init(optic)) == optic`.

!!! note
    Does not perform optic normalisation on the input. You may wish to call
    `normalise(optic)` before using this function if the optic you are passing
    was not obtained from a VarName.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._init(Accessors.@o _.a.b.c)
(@o _.a.b)

julia> AbstractPPL._init(Accessors.@o _[1][2][3])
(@o _[1][2])

julia> AbstractPPL._init(Accessors.@o _.a)
identity (generic function with 1 method)

julia> AbstractPPL._init(Accessors.@o _[1])
identity (generic function with 1 method)

julia> AbstractPPL._init(Accessors.@o _)
identity (generic function with 1 method)
"""
# This one needs normalise because it's going 'against' the direction of the
# linked list (otherwise you will end up with identities scattered throughout)
function _init(o::ComposedFunction{Outer,Inner}) where {Outer,Inner}
    return normalise(_init(o.outer) ∘ o.inner)
end
_init(::Accessors.PropertyLens) = identity
_init(::Accessors.IndexLens) = identity
_init(::typeof(identity)) = identity
