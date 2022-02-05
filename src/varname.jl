using Setfield
using Setfield: PropertyLens, ComposedLens, IdentityLens, IndexLens, DynamicIndexLens

"""
    VarName{sym}(lens::Lens=IdentityLens())

A variable identifier for a symbol `sym` and lens `lens`.

The Julia variable in the model corresponding to `sym` can refer to a single value or to a
hierarchical array structure of univariate, multivariate or matrix variables. The field `lens`
stores the indices requires to access the random variable from the Julia variable indicated by `sym`
as a tuple of tuples. Each element of the tuple thereby contains the indices of one lens
operation.

`VarName`s can be manually constructed using the `VarName{sym}(lens)` constructor, or from an
lens expression through the [`@varname`](@ref) convenience macro.

# Examples

```jldoctest; setup=:(using Setfield)
julia> vn = VarName{:x}(Setfield.IndexLens((Colon(), 1)) ∘ Setfield.IndexLens((2, )))
x[:,1][2]

julia> getlens(vn)
(@lens _[Colon(), 1][2])

julia> @varname x[:, 1][1+1]
x[:,1][2]
```
"""
struct VarName{sym,T<:Lens}
    lens::T

    function VarName{sym}(lens=IdentityLens()) where {sym}
        # TODO: Should we completely disallow or just `@warn` of limited support?
        if !is_static_lens(lens)
            error("attempted to construct `VarName` with dynamic lens of type $(nameof(typeof(lens)))")
        end
        return new{sym,typeof(lens)}(lens)
    end
end

"""
    is_static_lens(l::Lens)

Return `true` if `l` does not require runtime information to be resolved.

In particular it returns `false` for `Setfield.DynamicLens` and `Setfield.FunctionLens`.
"""
is_static_lens(l::Lens) = is_static_lens(typeof(l))
is_static_lens(::Type{<:Lens}) = false
is_static_lens(::Type{<:Union{PropertyLens, IndexLens, IdentityLens}}) = true
function is_static_lens(::Type{ComposedLens{LO, LI}}) where {LO, LI}
    return is_static_lens(LO) && is_static_lens(LI)
end

# A bit of backwards compatibility.
VarName{sym}(indexing::Tuple) where {sym} = VarName{sym}(tupleindex2lens(indexing))

"""
    VarName(vn::VarName, lens::Lens)
    VarName(vn::VarName, indexing::Tuple)

Return a copy of `vn` with a new index `lens`/`indexing`.

```jldoctest; setup=:(using Setfield)
julia> VarName(@varname(x[1][2:3]), Setfield.IndexLens((2,)))
x[2]

julia> VarName(@varname(x[1][2:3]), ((2,),))
x[2]

julia> VarName(@varname(x[1][2:3]))
x
```
"""
VarName(vn::VarName, lens::Lens = IdentityLens()) = VarName{getsym(vn)}(lens)

function VarName(vn::VarName, indexing::Tuple)
    return VarName{getsym(vn)}(tupleindex2lens(indexing))
end

tupleindex2lens(indexing::Tuple{}) = IdentityLens()
tupleindex2lens(indexing::Tuple{<:Tuple}) = IndexLens(first(indexing))
function tupleindex2lens(indexing::Tuple)
    return IndexLens(first(indexing)) ∘ tupleindex2lens(indexing[2:end])
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
getsym(vn::VarName{sym}) where {sym} = sym

"""
    getlens(vn::VarName)

Return the lens of the Julia variable used to generate `vn`.

## Examples

```jldoctest
julia> getlens(@varname(x[1][2:3]))
(@lens _[1][2:3])

julia> getlens(@varname(y))
(@lens _)
```
"""
getlens(vn::VarName) = vn.lens


"""
    get(obj, vn::VarName{sym})

Alias for `get(obj, PropertyLens{sym}() ∘ getlens(vn))`.
"""
function Setfield.get(obj, vn::VarName{sym}) where {sym}
    return Setfield.get(obj, PropertyLens{sym}() ∘ getlens(vn))
end

"""
    set(obj, vn::VarName{sym}, value)

Alias for `set(obj, PropertyLens{sym}() ∘ getlens(vn), value)`.
"""
function Setfield.set(obj, vn::VarName{sym}, value) where {sym}
    return Setfield.set(obj, PropertyLens{sym}() ∘ getlens(vn), value)
end


Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getlens(vn)), h)
function Base.:(==)(x::VarName, y::VarName)
    return getsym(x) == getsym(y) && getlens(x) == getlens(y)
end

# Allow compositions with lenses.
function Base.:∘(vn::VarName{sym,<:Lens}, lens::Lens) where {sym}
    return VarName{sym}(getlens(vn) ∘ lens)
end

function Base.show(io::IO, vn::VarName{<:Any,<:Lens})
    # No need to check `Setfield.has_atlens_support` since
    # `VarName` does not allow dynamic lenses.
    print(io, getsym(vn))
    _print_application(io, getlens(vn))
end

# This is all just to allow to convert `Colon()` into `:`.
_print_application(io::IO, l::Lens) = Setfield.print_application(io, l)
function _print_application(io::IO, l::ComposedLens)
    _print_application(io, l.outer)
    _print_application(io, l.inner)
end
_print_application(io::IO, l::IndexLens) =
    print(io, "[", join(map(prettify_index, l.indices), ","), "]")
# This is a bit weird but whatever. We're almost always going to
# `concretize` anyways.
_print_application(io::IO, l::DynamicIndexLens) = print(io, l, "(_)")

prettify_index(x) = string(x)
prettify_index(::Colon) = ":"

"""
    Symbol(vn::VarName)

Return a `Symbol` represenation of the variable identifier `VarName`.

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
    inspace(vn::Union{VarName, Symbol}, space::Tuple)

Check whether `vn`'s variable symbol is in `space`.  The empty tuple counts as the "universal space"
containing all variables. Subsumption (see [`subsume`](@ref)) is respected.

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
    return getsym(u) == getsym(v) && subsumes(u.lens, v.lens)
end

# Idea behind `subsumes` for `Lens` is that we traverse the two lenses in parallel,
# checking `subsumes` for every level. This for example means that if we are comparing
# `PropertyLens{:a}` and `PropertyLens{:b}` we immediately know that they do not subsume
# each other since at the same level/depth they access different properties.
# E.g. `x`, `x[1]`, i.e. `u` is always subsumed by `t`
subsumes(::IdentityLens, ::IdentityLens) = true
subsumes(::IdentityLens, ::Lens) = true
subsumes(::Lens, ::IdentityLens) = false

subsumes(t::ComposedLens, u::ComposedLens) =
    subsumes(t.outer, u.outer) && subsumes(t.inner, u.inner)

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
subsumes(
    t::Union{IndexLens,ComposedLens{<:IndexLens}},
    u::Union{IndexLens,ComposedLens{<:IndexLens}}
) = subsumes_index(t, u)


const subsumedby = (t, u) -> subsumes(u, t)
const ⊒ = subsumes
const ⊑ = subsumedby
const ⋣ = !subsumes
const ⋢ = !subsumedby
const ≍ = (t, u) -> t ⋢ u && u ⋢ t

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
```jldoctest; setup=:(using Setfield; using AbstractPPL: subsumes_index)
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
combine_indices(lens::IndexLens) = (lens.indices,), nothing
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
function subsumes_index(t1::Tuple, t2::Tuple)  # does x[i]... subsume x[j]...?
    first_subsumed = all(zip(first(t1), first(t2))) do (i, j)
        if j isa Colon
            error("Colons cannot be subsumed")
        elseif i isa Colon
            return true
        else
            return issubset(j, i)
        end
    end
    return first_subsumed && subsumes_index(Base.tail(t1), Base.tail(t2))
end


"""
    concretize_index(original_index, lowered_index)

Create the index to be emitted in `concretize`.  `original_index` is the original, unconcretized
index, and `lowered_index` the respective position of the result of `to_indices`.

The only purpose of this are special cases like `:`, which we want to avoid becoming a
`Base.Slice(OneTo(...))` -- it would confuse people when printed.  Instead, we concretize to a
`UnitRange` based on the `lowered_index`, just what you'd get with an explicit `begin:end`
"""
reconcretize_index(original_index, lowered_index) = lowered_index
reconcretize_index(original_index::Colon, lowered_index::Base.Slice) = (UnitRange(lowered_index))

"""
    concretize(l::Lens, x)

Return `l` instantiated on `x`, i.e. any information related to the runtime shape of `x` is
evaluated.  This concerns `begin`, `end`, and `:` slices.

Basically, every index is converted to a concrete value using `Base.to_index` on `x`.  However, `:`
slices are only converted to `UnitRange`s (`a:b`) (as opposed to `Base.Slice{Base.OneTo}`), to keep
the result close to the original indexing.
"""
concretize(I::Lens, x) = I
concretize(I::DynamicIndexLens, x) = concretize(IndexLens(I.f(x)), x)
concretize(I::IndexLens, x) = IndexLens(map(reconcretize_index, I.indices, to_indices(x, I.indices)))
function concretize(I::ComposedLens, x)
    x_inner = get(x, I.outer) # TODO: get view here
    return ComposedLens(concretize(I.outer, x), concretize(I.inner, x_inner))
end

"""
    concretize(vn::VarName, x)

Return `vn` concretized on `x`, i.e. any information related to the runtime shape of `x` is
evaluated. This concerns `begin`, `end`, and `:` slices.

# Examples
```jldoctest; setup=:(using Setfield)
julia> x = (a = [1.0 2.0; 3.0 4.0; 5.0 6.0], );

julia> getlens(@varname(x.a[1:end, end][:], true)) # concrete=true required for @varname
(@lens _.a[1:3, 2][1:3])

julia> y = zeros(10, 10);

julia> @varname(y[:], true)
y[1:100]

julia> AbstractPPL.getlens(AbstractPPL.concretize(@varname(y[:]), y)).indices
(1:100,)
```
"""
concretize(vn::VarName, x) = VarName(vn, concretize(getlens(vn), x))

"""
    @varname(expr, concretize=false)

A macro that returns an instance of [`VarName`](@ref) given a symbol or indexing expression `expr`.

If `concretize` is `true`, the resulting expression will be wrapped in a [`concretize`](@ref) call.

Note that expressions involving dynamic indexing, i.e. `begin` and/or `end`, will always need to be
concretized as `VarName` only supports non-dynamic indexing as determined by
[`is_static_index`](@ref). See examples below.

## Examples
### Dynamic indexing
```jldoctest
julia> x = (a = [1.0 2.0; 3.0 4.0; 5.0 6.0], );

julia> @varname(x.a[1:end, end][:], true)
x.a[1:3,2][1:3]

julia> @varname(x.a[end])
ERROR: LoadError: Variable name `x.a[end]` is dynamic and requires concretization!
[...]

julia> @varname(x.a[end], true)
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

Under the hood Setfield.jl's `Lens` are used for the indexing:

```jldoctest
julia> getlens(@varname(x))
(@lens _)

julia> getlens(@varname(x[1]))
(@lens _[1])

julia> getlens(@varname(x[:, 1]))
(@lens _[Colon(), 1])

julia> getlens(@varname(x[:, 1][2]))
(@lens _[Colon(), 1][2])

julia> getlens(@varname(x[1,2][1+5][45][3]))
(@lens _[1, 2][6][45][3])
```

This also means that we support property access:

```jldoctest
julia> getlens(@varname(x.a))
(@lens _.a)

julia> getlens(@varname(x.a[1]))
(@lens _.a[1])

julia> x = (a = [(b = rand(2), )], ); getlens(@varname(x.a[1].b[end], true))
(@lens _.a[1].b[2])
```

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro varname(expr::Union{Expr,Symbol}, concretize::Bool=false)
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

        if concretize
            return :(
                $(AbstractPPL.VarName){$(QuoteNode(sym))}(
                    $(AbstractPPL.concretize)($lens, $sym_escaped)
                )
            )
        elseif !concretize && Setfield.need_dynamic_lens(expr)
            error("Variable name `$(expr)` is dynamic and requires concretization!")
        else
            :($(AbstractPPL.VarName){$(QuoteNode(sym))}($lens))
        end
    else
        error("Malformed variable name `$(expr)`!")
    end
end

drop_escape(x) = x
function drop_escape(expr::Expr)
    Meta.isexpr(expr, :escape) && return drop_escape(expr.args[1])
    return Expr(expr.head, map(x -> drop_escape(x), expr.args)...)
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
