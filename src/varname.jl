using Setfield
using Setfield: PropertyLens, ComposedLens, IdentityLens, IndexLens, DynamicIndexLens

"""
    VarName{sym}(indexing::Tuple=())

A variable identifier for a symbol `sym` and indices `indexing` in the format
returned by [`@vinds`](@ref).

The Julia variable in the model corresponding to `sym` can refer to a single value or to a
hierarchical array structure of univariate, multivariate or matrix variables. The field `indexing`
stores the indices requires to access the random variable from the Julia variable indicated by `sym`
as a tuple of tuples. Each element of the tuple thereby contains the indices of one indexing
operation.

`VarName`s can be manually constructed using the `VarName{sym}(indexing)` constructor, or from an
indexing expression through the [`@varname`](@ref) convenience macro.

# Examples

```jldoctest; setup=:(using Setfield)
julia> vn = VarName{:x}(Setfield.IndexLens((Colon(), 1)) ∘ Setfield.IndexLens((2, )))
x[:,1][2]

julia> vn.indexing
((Colon(), 1), (2,))

julia> @varname x[:, 1][1+1]
x[:,1][2]
```
"""
struct VarName{sym, T<:Lens}
    indexing::T

    VarName{sym}(indexing=IdentityLens()) where {sym} = new{sym,typeof(indexing)}(indexing)
end

"""
    VarName(vn::VarName, indexing=())

Return a copy of `vn` with a new index `indexing`.

```jldoctest
julia> VarName(@varname(x[1][2:3]), ((2,),))
x[2]

julia> VarName(@varname(x[1][2:3]))
x
```
"""
function VarName(vn::VarName, indexing=IdentityLens())
    return VarName{getsym(vn)}(indexing)
end


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
getsym(vn::VarName{sym}) where sym = sym


"""
    getindexing(vn::VarName)

Return the indexing tuple of the Julia variable used to generate `vn`.

## Examples

```jldoctest
julia> getindexing(@varname(x[1][2:3]))
(@lens _[1][2:3])

julia> getindexing(@varname(y))
()
```
"""
getindexing(vn::VarName) = vn.indexing


Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getindexing(vn)), h)
Base.:(==)(x::VarName, y::VarName) = getsym(x) == getsym(y) && getindexing(x) == getindexing(y)

# Composition rules similar to the standard one for lenses, but we need a special
# one for the "empty" `VarName{..., Tuple{}}`.
Base.:∘(vn::VarName{sym,<:IdentityLens}, lens::Lens) where {sym} = VarName{sym}(lens)
Base.:∘(vn::VarName{sym,<:Lens}, lens::Lens) where {sym} = VarName{sym}(vn.indexing ∘ lens)

function Base.show(io::IO, vn::VarName{<:Any, <:Tuple})
    print(io, getsym(vn))
    for indices in getindexing(vn)
        print(io, "[")
        join(io, map(prettify_index, indices), ",")
        print(io, "]")
    end
end

function Base.show(io::IO, vn::VarName{<:Any, <:Lens})
    print(io, getsym(vn))
    _print_application(io, vn.indexing)
end

_print_application(io::IO, l::Lens) = Setfield.print_application(io, l)
function _print_application(io::IO, l::ComposedLens)
    _print_application(io, l.outer)
    _print_application(io, l.inner)
end
_print_application(io::IO, l::IndexLens) = print(io, "[", join(map(prettify_index, l.indices), ","), "]")
# This is a bit weird but whatever. We're almost always going to
# `concretize` anyways.
_print_application(io::IO, l::DynamicIndexLens) = print(io, l, "(_)")

prettify_index(x) = string(x)
prettify_index(::Colon) = ":"

"""
    Symbol(vn::VarName)

Return a `Symbol` represenation of the variable identifier `VarName`.

```jldoctest
julia> Symbol(@varname(x[1][2:3]))
Symbol("x[1][2:3]")
```
"""
Base.Symbol(vn::VarName) = Symbol(string(vn))  # simplified symbol


"""
    inspace(vn::Union{VarName, Symbol}, space::Tuple)

Check whether `vn`'s variable symbol is in `space`.  The empty tuple counts as the "universal space"
containing all variables.  Subsumption (see [`subsume`](@ref)) is respected.

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
    return getsym(u) == getsym(v) && subsumes(u.indexing, v.indexing)
end

# Idea behind `subsumes` for `Lens` is that we traverse the two lenses in parallel,
# checking `subsumes` for every level. This for example means that if we are comparing
# `PropertyLens{:a}` and `PropertyLens{:b}` we immediately know that they do not subsume
# each other since at the same level/depth they access different properties.
# E.g. `x`, `x[1]`, i.e. `u` is always subsumed by `t`
subsumes(t::IdentityLens, u::Lens) = true
subsumes(t::Lens, u::IdentityLens) = false

subsumes(t::ComposedLens, u::ComposedLens) = subsumes(t.outer, u.outer) && subsumes(t.inner, u.inner)

# If `t` is still a composed lens, then there is no way it can subsume `u` since `u` is a
# leaf of the "lens-tree".
subsumes(t::ComposedLens, u::PropertyLens) = false
# Here we need to check if `u.outer` (i.e. the next lens to be applied from `u`) is
# subsumed by `t`, since this would mean that the rest of the composition is also subsumed
# by `t`.
subsumes(t::PropertyLens, u::ComposedLens) = subsumes(t, u.outer)

# For `PropertyLens` either they have the same `name` and thus they are indeed the same.
subsumes(t::PropertyLens{name}, u::PropertyLens{name}) where {name} = true
# Otherwise they represent different properties, and thus are not the same.
subsumes(t::PropertyLens, u::PropertyLens) = false

# Indices subsumes if they are subindices, i.e. we just call `_issubindex`.
# FIXME: Does not support `DynamicIndexLens`.
# FIXME: Does not correctly handle cases such as `subsumes(x, x[:])`
#        (but neither did old implementation).
subsumes(t::IndexLens, u::IndexLens) = _issubindex(t.indices, u.indices)
subsumes(t::ComposedLens{<:IndexLens}, u::ComposedLens{<:IndexLens}) = subsumes_index(t, u)
subsumes(t::IndexLens, u::ComposedLens{<:IndexLens}) = subsumes_index(t, u)
subsumes(t::ComposedLens{<:IndexLens}, u::IndexLens) = subsumes_index(t, u)

# Since expressions such as `x[:][:][:][1]` and `x[1]` are equal,
# the indexing behavior must be considered jointly.
# Therefore we must recurse until we reach something that is NOT
# indexing, and then consider the sequence of indices leading up to this.
"""
    subsumes_index(t::Lens, u::Lens)

Return `true` if the indexing represented by `t` subsumes `u`.

This is mostly useful for comparing compositions involving `IndexLens`
e.g. `_[1][2].a[2]` and `_[1][2].a`. In such a scenario we do the following:
1. Combine `[1][2]` into a `Tuple` of indices using [`combine_indices`](@ref).
2. Do the same for `[1][2]`.
3. Compare the two tuples from (1) and (2) using `subsumes_index`.
4. Since we're still undecided, we call `subsume(@lens(_.a[2]), @lens(_.a))`
   which then returns `false`.

# Example
```jldoctest; setup=:(using Setfield)
julia> t = @lens(_[1].a); u = @lens(_[1]);

julia> subsumes_index(t, u)
false

julia> subsumes_index(u, t)
true

julia> # `IdentityLens` subsumes all.
       subsumes_index(@lens(_), t)
true

julia> # None subsumes `IdentityLens`.
       subsumes_index(t, @lens(_))
false

julia> AbstractPPL.subsumes(@lens(_[1][2].a[2]), @lens(_[1][2].a))
false

julia> AbstractPPL.subsumes(@lens(_[1][2].a), @lens(_[1][2].a[2]))
true
```
"""
function subsumes_index(t::Lens, u::Lens)
    t_indices, t_next = combine_indices(t)
    u_indices, u_next = combine_indices(u)

    # If we already know that `u` is not subsumed by `t`, return early.
    if !subsumes_index(t_indices, u_indices)
        return false
    end

    if t_next === nothing
        # Means that there's nothing left for `t` and either nothing
        # or something left for `u`, i.e. `t` indeed `subsumes` `u`.
        return true
    elseif u_next === nothing
        # If `t_next` is not `nothing` but `u_ntext` is, then
        # `t` does not subsume `u`.
        return false
    end

    # If neither is `nothing` we continue.
    return subsumes(t_next, u_next)
end

"""
    combine_indices(lens)

Return sequential indexing into a single `Tuple` of indices,
e.g. `x[:][1][2]` becomes `((Colon(), ), (1, ), (2, ))`.

The result is compatible with [`subsumes_index`](@ref) for `Tuple` input.
"""
combine_indices(lens::Lens) = (), lens
combine_indices(lens::IndexLens) = (lens.indices, ), nothing
function combine_indices(lens::ComposedLens{<:IndexLens})
    indices, next = combine_indices(lens.inner)
    return (lens.outer.indices, indices...), next
end

"""
    subsumes_index(left_index::Tuple, right_index::Tuple)

Return `true` if `right_index` is subsumed by `left_index`.

Currently _not_ supported are: 
- Boolean indexing, literal `CartesianIndex` (these could be added, though)
- Linear indexing of multidimensional arrays: `x[4]` does not subsume `x[2, 2]` for a matrix `x`
- Trailing ones: `x[2, 1]` does not subsume `x[2]` for a vector `x`
- Dynamic indexing, e.g. `x[1]` does not subsume `x[begin]`.
"""
subsumes_index(::Tuple{}, ::Tuple{}) = true  # x subsumes x
subsumes_index(::Tuple{}, ::Tuple) = true    # x subsumes x[1]
subsumes_index(::Tuple, ::Tuple{}) = false   # x[1] does not subsume x
function subsumes_index(t::Tuple, u::Tuple)  # does x[i]... subsume x[j]...?
    return _issubindex(first(t), first(u)) && subsumes_index(Base.tail(t), Base.tail(u))
end

const AnyIndex = Union{Int, AbstractVector{Int}, Colon} 
_issubindex_(::Tuple{Vararg{AnyIndex}}, ::Tuple{Vararg{AnyIndex}}) = false
function _issubindex(t::NTuple{N, AnyIndex}, u::NTuple{N, AnyIndex}) where {N}
    return all(_issubrange(j, i) for (i, j) in zip(t, u))
end

const ConcreteIndex = Union{Int, AbstractVector{Int}} # this include all kinds of ranges

"""Determine whether indices `i` are contained in `j`, treating `:` as universal set."""
_issubrange(i::ConcreteIndex, j::ConcreteIndex) = issubset(i, j)
_issubrange(i::Colon, j::Colon) = true
_issubrange(i::Colon, j::ConcreteIndex) = false
# FIXME: [2021-07-31] This is wrong but we have tests in DPPL that tell
# us that it SHOULD be correct. I'll leave it as is for now to ensure that
# we preserve the status quo, but I'm confused.
_issubrange(i::ConcreteIndex, j::Colon) = false

"""
    concretize(l::Lens, x)

Return `l` instantiated on `x`, i.e. any runtime information evaluated using `x`.
"""

concretize(I::Lens, x) = I
concretize(I::DynamicIndexLens, x) = IndexLens(I.f(x))
function concretize(I::ComposedLens, x)
    x_inner = get(x, I.outer)
    return ComposedLens(concretize(I.outer, x), concretize(I.inner, x_inner))
end
"""
    concretize(vn::VarName, x)

Return `vn` instantiated on `x`, i.e. any runtime information evaluated using `x`.

# Examples
```jldoctest
julia> x = (a = [1.0 2.0;], );

julia> vn = @varname(x.a[1, :])
x.a[1,:]

julia> AbstractPPL.concretize(vn, x)
x.a[1,:]

julia> vn = @varname(x.a[1, end][:]);

julia> AbstractPPL.concretize(vn, x)
x.a[1,2][:]
```
"""
concretize(vn::VarName, x) = VarName(vn, concretize(vn.indexing, x))

"""
    @varname(expr[, concretize])

A macro that returns an instance of [`VarName`](@ref) given a symbol or indexing expression `expr`.

If `concretize` is `true`, the resulting expression will be wrapped in a [`concretize`](@ref) call.
This is useful if you for example want to ensure that no `Setfield.DynamicLens` is used.

The `sym` value is taken from the actual variable name, and the index values are put appropriately
into the constructor (and resolved at runtime).

## Examples

```jldoctest
julia> @varname(x).indexing
()

julia> @varname(x[1]).indexing
(@lens _[1])

julia> @varname(x[:, 1]).indexing
(@lens _[Colon(), 1])

julia> @varname(x[:, 1][2]).indexing
(@lens _[Colon(), 1][2])

julia> @varname(x[1,2][1+5][45][3]).indexing
(@lens _[1, 2][6][45][3])
```

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro varname(expr::Union{Expr, Symbol}, concretize=false)
    return varname(expr, concretize)
end

varname(sym::Symbol, concretize=false) = :($(AbstractPPL.VarName){$(QuoteNode(sym))}())
function varname(expr::Expr, concretize=false)
    if Meta.isexpr(expr, :ref) || Meta.isexpr(expr, :.)
        # Split into object/base symbol and lens.
        sym_escaped, lens = Setfield.parse_obj_lens(expr)
        # Setfield.jl escapes the return symbol, so we need to unescape
        # to call `QuoteNode` on it.
        sym = drop_escape(sym_escaped)

        return if Setfield.need_dynamic_lens(expr)
            :($(AbstractPPL.concretize)($(AbstractPPL.VarName){$(QuoteNode(sym))}($lens), $sym_escaped))
        else
            :($(AbstractPPL.VarName){$(QuoteNode(sym))}($lens))
        end
    else
        error("Malformed variable name $(expr)!")
    end
end

drop_escape(x) = x
function drop_escape(expr::Expr)
    Meta.isexpr(expr, :escape) && return drop_escape(expr.args[1])
    return Expr(expr.head, map(x -> drop_escape(x), expr.args)...)
end

@static if VERSION ≥ v"1.5.0-DEV.666"
    # TODO: Replace once https://github.com/jw3126/Setfield.jl/pull/155 has been merged.
    function Setfield.lower_index(collection::Symbol, index, dim)
        if Setfield.isexpr(index, :call)
            return Expr(:call, Setfield.lower_index.(collection, index.args, dim)...)
        elseif (index === :end)
            if dim === nothing
                return :($(Base.lastindex)($collection))
            else
                return :($(Base.lastindex)($collection, $dim))
            end
        elseif index === :begin
            if dim === nothing
                return :($(Base.firstindex)($collection))
            else
                return :($(Base.firstindex)($collection, $dim))
            end
        end
        return index
    end

    function Setfield.need_dynamic_lens(ex)
        return Setfield.foldtree(false, ex) do yes, x
            (yes || x === :end || x === :begin || x === :_)
        end
    end
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
macro vsym(expr::Union{Expr, Symbol})
    return QuoteNode(vsym(expr))
end

"""
    vsym(expr)

Return name part of the [`@varname`](@ref)-compatible expression `expr` as a symbol for input of the
[`VarName`](@ref) constructor."""
function vsym end

vsym(expr::Symbol) = expr
function vsym(expr::Expr)
    if Meta.isexpr(expr, :ref) || Meta.isexpr(expr, :.)
        return vsym(expr.args[1])
    else
        error("Malformed variable name $(expr)!")
    end
end

