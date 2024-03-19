using Accessors
using Accessors: ComposedOptic, PropertyLens, IndexLens, DynamicIndexLens

using Setfield: Setfield
using MacroTools

const ALLOWED_OPTICS = Union{typeof(identity),PropertyLens,IndexLens,ComposedOptic}

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
struct VarName{sym,T}
    optic::T

    function VarName{sym}(optic=identity) where {sym}
        if !is_static_optic(typeof(optic))
            throw(ArgumentError("attempted to construct `VarName` with unsupported optic of type $(nameof(typeof(optic)))"))
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
function is_static_optic(::Type{ComposedOptic{LO,LI}}) where {LO,LI}
    return is_static_optic(LO) && is_static_optic(LI)
end
is_static_optic(::Type{<:DynamicIndexLens}) = false

# A bit of backwards compatibility.
VarName{sym}(indexing::Tuple) where {sym} = VarName{sym}(tupleindex2optic(indexing))

"""
    VarName(vn::VarName, optic)
    VarName(vn::VarName, indexing::Tuple)

Return a copy of `vn` with a new index `optic`/`indexing`.

```jldoctest; setup=:(using Accessors)
julia> VarName(@varname(x[1][2:3]), Accessors.IndexLens((2,)))
x[2]

julia> VarName(@varname(x[1][2:3]), ((2,),))
x[2]

julia> VarName(@varname(x[1][2:3]))
x
```
"""
VarName(vn::VarName, optic=identity) = VarName{getsym(vn)}(optic)

function VarName(vn::VarName, indexing::Tuple)
    return VarName{getsym(vn)}(tupleindex2optic(indexing))
end

tupleindex2optic(indexing::Tuple{}) = identity
tupleindex2optic(indexing::Tuple{<:Tuple}) = IndexLens(first(indexing)) # TODO: rest?
function tupleindex2optic(indexing::Tuple)
    return IndexLens(first(indexing)) ∘ tupleindex2optic(indexing[2:end])
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

Alias for `getoptic(vn)(obj)`.
"""
function Base.get(obj, vn::VarName{sym}) where {sym}
    return getoptic(vn)(obj)
end

"""
    set(obj, vn::VarName{sym}, value)

Alias for `set(obj, PropertyLens{sym}() ∘ getoptic(vn), value)`.
"""
function Accessors.set(obj, vn::VarName{sym}, value) where {sym}
    return Accessors.set(obj, PropertyLens{sym}() ∘ getoptic(vn), value)
end


Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getoptic(vn)), h)
function Base.:(==)(x::VarName, y::VarName)
    return getsym(x) == getsym(y) && getoptic(x) == getoptic(y)
end

function Base.show(io::IO, vn::VarName{sym,T}) where {sym,T}
    print(io, getsym(vn))
    _show_optic(io, getoptic(vn))
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
    print(io, shortstr)
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
    return getsym(u) == getsym(v) && subsumes(getoptic(u), getoptic(v))
end

# Idea behind `subsumes` for `Lens` is that we traverse the two lenses in parallel,
# checking `subsumes` for every level. This for example means that if we are comparing
# `PropertyLens{:a}` and `PropertyLens{:b}` we immediately know that they do not subsume
# each other since at the same level/depth they access different properties.
# E.g. `x`, `x[1]`, i.e. `u` is always subsumed by `t`
subsumes(::typeof(identity), ::typeof(identity)) = true
subsumes(::typeof(identity), ::ALLOWED_OPTICS) = true
subsumes(::ALLOWED_OPTICS, ::typeof(identity)) = false

subsumes(t::ComposedOptic, u::ComposedOptic) =
    subsumes(t.outer, u.outer) && subsumes(t.inner, u.inner)

# If `t` is still a composed lens, then there is no way it can subsume `u` since `u` is a
# leaf of the "lens-tree".
subsumes(t::ComposedOptic, u::PropertyLens) = false
# Here we need to check if `u.outer` (i.e. the next lens to be applied from `u`) is
# subsumed by `t`, since this would mean that the rest of the composition is also subsumed
# by `t`.
subsumes(t::PropertyLens, u::ComposedOptic) = subsumes(t, u.inner)

# For `PropertyLens` either they have the same `name` and thus they are indeed the same.
subsumes(t::PropertyLens{name}, u::PropertyLens{name}) where {name} = true
# Otherwise they represent different properties, and thus are not the same.
subsumes(t::PropertyLens, u::PropertyLens) = false

# Indices subsumes if they are subindices, i.e. we just call `_issubindex`.
# FIXME: Does not support `DynamicIndexLens`.
# FIXME: Does not correctly handle cases such as `subsumes(x, x[:])`
#        (but neither did old implementation).
subsumes(
    t::Union{IndexLens,ComposedOptic{<:ALLOWED_OPTICS,<:IndexLens}},
    u::Union{IndexLens,ComposedOptic{<:ALLOWED_OPTICS,<:IndexLens}}
) = subsumes_indices(t, u)


subsumedby(t, u) = subsumes(u, t)
uncomparable(t, u) = t ⋢ u && u ⋢ t
const ⊒ = subsumes
const ⊑ = subsumedby
const ⋣ = !subsumes
const ⋢ = !subsumedby
const ≍ = uncomparable

# Since expressions such as `x[:][:][:][1]` and `x[1]` are equal,
# the indexing behavior must be considered jointly.
# Therefore we must recurse until we reach something that is NOT
# indexing, and then consider the sequence of indices leading up to this.
"""
    subsumes_indices(t, u)

Return `true` if the indexing represented by `t` subsumes `u`.

This is mostly useful for comparing compositions involving `IndexLens`
e.g. `_[1][2].a[2]` and `_[1][2].a`. In such a scenario we do the following:
1. Combine `[1][2]` into a `Tuple` of indices using [`combine_indices`](@ref).
2. Do the same for `[1][2]`.
3. Compare the two tuples from (1) and (2) using `subsumes_indices`.
4. Since we're still undecided, we call `subsume(@o(_.a[2]), @o(_.a))`
   which then returns `false`.

# Example
```jldoctest; setup=:(using Accessors; using AbstractPPL: subsumes_indices)
julia> t = @o(_[1].a); u = @o(_[1]);

julia> subsumes_indices(t, u)
false

julia> subsumes_indices(u, t)
true

julia> # `identity` subsumes all.
       subsumes_indices(identity, t)
true

julia> # None subsumes `identity`.
       subsumes_indices(t, identity)
false

julia> AbstractPPL.subsumes(@o(_[1][2].a[2]), @o(_[1][2].a))
false

julia> AbstractPPL.subsumes(@o(_[1][2].a), @o(_[1][2].a[2]))
true
```
"""
function subsumes_indices(t::ALLOWED_OPTICS, u::ALLOWED_OPTICS)
    t_indices, t_next = combine_indices(t)
    u_indices, u_next = combine_indices(u)

    # If we already know that `u` is not subsumed by `t`, return early.
    if !subsumes_indices(t_indices, u_indices)
        return false
    end

    if t_next === nothing
        # Means that there's nothing left for `t` and either nothing
        # or something left for `u`, i.e. `t` indeed `subsumes` `u`.
        return true
    elseif u_next === nothing
        # If `t_next` is not `nothing` but `u_next` is, then
        # `t` does not subsume `u`.
        return false
    end

    # If neither is `nothing` we continue.
    return subsumes(t_next, u_next)
end

"""
    combine_indices(optic)

Return sequential indexing into a single `Tuple` of indices,
e.g. `x[:][1][2]` becomes `((Colon(), ), (1, ), (2, ))`.

The result is compatible with [`subsumes_indices`](@ref) for `Tuple` input.
"""
combine_indices(optic::ALLOWED_OPTICS) = (), optic
combine_indices(optic::IndexLens) = (optic.indices,), nothing
function combine_indices(optic::ComposedOptic{<:ALLOWED_OPTICS,<:IndexLens})
    indices, next = combine_indices(optic.outer)
    return (optic.inner.indices, indices...), next
end

"""
    subsumes_indices(left_indices::Tuple, right_indices::Tuple)

Return `true` if `right_indices` is subsumed by `left_indices`.  `left_indices` is assumed to be 
concretized and consist of either `Int`s or `AbstractArray`s of scalar indices that are supported 
by array A.

Currently _not_ supported are: 
- Boolean indexing, literal `CartesianIndex` (these could be added, though)
- Linear indexing of multidimensional arrays: `x[4]` does not subsume `x[2, 2]` for a matrix `x`
- Trailing ones: `x[2, 1]` does not subsume `x[2]` for a vector `x`
"""
subsumes_indices(::Tuple{}, ::Tuple{}) = true  # x subsumes x
subsumes_indices(::Tuple{}, ::Tuple) = true    # x subsumes x...
subsumes_indices(::Tuple, ::Tuple{}) = false   # x... does not subsume x
function subsumes_indices(t1::Tuple, t2::Tuple)  # does x[i]... subsume x[j]...?
    first_subsumed = all(Base.splat(subsumes_index), zip(first(t1), first(t2)))
    return first_subsumed && subsumes_indices(Base.tail(t1), Base.tail(t2))
end

subsumes_index(i::Colon, ::Colon) = error("Colons cannot be subsumed")
subsumes_index(i, ::Colon) = error("Colons cannot be subsumed")
# Necessary to avoid ambiguity errors.
subsumes_index(::AbstractVector, ::Colon) = error("Colons cannot be subsumed")
subsumes_index(i::Colon, j) = true
subsumes_index(i::AbstractVector, j) = issubset(j, i)
subsumes_index(i, j) = i == j


"""
    ConcretizedSlice(::Base.Slice)

An indexing object wrapping the range of a `Base.Slice` object representing the concrete indices a
`:` indicates.  Behaves the same, but prints differently, namely, still as `:`.
"""
struct ConcretizedSlice{T,R} <: AbstractVector{T}
    range::R
end

ConcretizedSlice(s::Base.Slice{R}) where {R} = ConcretizedSlice{eltype(s.indices),R}(s.indices)
Base.show(io::IO, s::ConcretizedSlice) = print(io, ":")
Base.show(io::IO, ::MIME"text/plain", s::ConcretizedSlice) =
    print(io, "ConcretizedSlice(", s.range, ")")
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
reconcretize_index(original_index::Colon, lowered_index::Base.Slice) =
    ConcretizedSlice(lowered_index)

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
concretize(I::IndexLens, x) = IndexLens(reconcretize_index.(I.indices, to_indices(x, I.indices)))
function concretize(I::ComposedOptic, x)
    x_inner = I.inner(x) # TODO: get view here
    return ComposedOptic(concretize(I.outer, x_inner), concretize(I.inner, x))
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

julia> # The underlying value is conretized, though:
       AbstractPPL.getoptic(AbstractPPL.concretize(@varname(y[:]), y)).indices[1]
ConcretizedSlice(Base.OneTo(100))
```
"""
concretize(vn::VarName, x) = VarName(vn, concretize(getoptic(vn), x))

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

Interpolation can be used for names (only the base name as well as property name).  Variables within
indices are always evaluated in the calling scope

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
        sym_escaped, lens = Setfield.parse_obj_lens(expr)
        lens = lens_to_optic(lens)
        # Setfield.jl escapes the return symbol, so we need to unescape
        # to call `QuoteNode` on it.
        sym = drop_escape(sym_escaped)

        # This is to handle interpolated heads -- Setfield treats them differently:
        # julia> Setfield.parse_obj_lens(@q $name.a)
        # (:($(Expr(:escape, :_))), :((Setfield.compose)($(Expr(:escape, :name)), (Setfield.PropertyLens){:a}())))
        # julia> Setfield.parse_obj_lens(@q x.a)
        # (:($(Expr(:escape, :x))), :((Setfield.compose)((Setfield.PropertyLens){:a}())))
        if sym != :_
            sym = QuoteNode(sym)
        else
            sym = lens.args[2]
            lens = Expr(:call, lens.args[1], lens.args[3:end]...)
        end

        if concretize
            return :(
                $(AbstractPPL.VarName){$sym}(
                    $(AbstractPPL.concretize)($lens, $sym_escaped)
                )
            )
        elseif Setfield.need_dynamic_lens(expr)
            error("Variable name `$(expr)` is dynamic and requires concretization!")
        else
            :($(AbstractPPL.VarName){$sym}($lens))
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

function lens_to_optic(lens)
    MacroTools.postwalk(lens) do ex
        if ex == (Setfield.compose)
            return :(Accessors.opcompose)
        elseif ex == (Setfield.PropertyLens)
            return :(Accessors.PropertyLens)
        elseif ex == (Setfield.IndexLens)
            return :(Accessors.IndexLens)
        elseif ex == (Setfield.DynamicIndexLens)
            return :(Accessors.DynamicIndexLens)
        end
        return ex
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
