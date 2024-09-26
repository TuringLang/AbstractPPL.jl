using Accessors
using Accessors: ComposedOptic, PropertyLens, IndexLens, DynamicIndexLens
using StructTypes: StructTypes
using JSON3: JSON3

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
function Base.:∘(optic::ALLOWED_OPTICS, vn::VarName{sym,<:ALLOWED_OPTICS}) where {sym}
    vn_optic = getoptic(vn)
    if vn_optic == identity
        return VarName{sym}(optic)
    elseif optic == identity
        return vn
    else
        return VarName{sym}(optic ∘ vn_optic)
    end
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
containing all variables. Subsumption (see [`subsumes`](@ref)) is respected.

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
# Here we need to check if `u.inner` (i.e. the next lens to be applied from `u`) is
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


"""
    subsumedby(t, u)

True if `t` is subsumed by `u`, i.e., if `subsumes(u, t)` is true.
"""
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
ConcretizedSlice(s::Base.OneTo{R}) where {R} = ConcretizedSlice(Base.Slice(s))
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

julia> # The underlying value is concretized, though:
       AbstractPPL.getoptic(AbstractPPL.concretize(@varname(y[:]), y)).indices[1]
ConcretizedSlice(Base.OneTo(100))
```
"""
concretize(vn::VarName, x) = VarName(vn, concretize(getoptic(vn), x))

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
            return :(
                $(AbstractPPL.VarName){$sym}(
                $(AbstractPPL.concretize)($optics, $sym_escaped)
            )
            )
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
    obj, optic
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
            optics = :($(Accessors.DynamicIndexLens)($(esc(collection)) -> ($(lindices...),)))
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
            throw(ArgumentError(
                string("Error while parsing :($ex). Second argument to `getproperty` can only be",
                    "a `Symbol` or `String` literal, received `$property` instead.")
            ))
        end
    else
        obj = esc(ex)
        return obj, ()
    end
    obj, tuple(frontoptics..., optics)
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
    index_to_str(i)

Generates a string representation of the index `i`, or a tuple thereof.

## Examples

```jldoctest
julia> AbstractPPL.index_to_str(2)
"2"

julia> AbstractPPL.index_to_str((1, 2:5))
"(1, 2:5,)"

julia> AbstractPPL.index_to_str(:)
":"

julia> AbstractPPL.index_to_str(AbstractPPL.ConcretizedSlice(Base.Slice(Base.OneTo(10))))
"ConcretizedSlice(Base.OneTo(10))"
```
"""
index_to_str(i::Integer) = string(i)
index_to_str(r::UnitRange) = "$(first(r)):$(last(r))"
index_to_str(::Colon) = ":"
index_to_str(s::ConcretizedSlice{T,R}) where {T,R} = "ConcretizedSlice(" * repr(s.range) * ")"
index_to_str(t::Tuple) = "(" * join(map(index_to_str, t), ", ") * ",)"

"""
    optic_to_nt(optic)

Convert an optic to a named tuple representation.

## Examples
```jldoctest; setup=:(using Accessors)
julia> AbstractPPL.optic_to_nt(identity)
(type = "identity",)

julia> AbstractPPL.optic_to_nt(@optic _.a)
(type = "property", field = "a")

julia> AbstractPPL.optic_to_nt(@optic _.a.b)
(type = "composed", outer = (type = "property", field = "b"), inner = (type = "property", field = "a"))

julia> AbstractPPL.optic_to_nt(@optic _[1])  # uses index_to_str()
(type = "index", indices = "(1,)")
```
"""
optic_to_nt(::typeof(identity)) = (type = "identity",)
optic_to_nt(::PropertyLens{sym}) where {sym} = (type = "property", field = String(sym))
optic_to_nt(i::IndexLens) = (type = "index", indices = index_to_str(i.indices))
optic_to_nt(c::ComposedOptic) = (type = "composed", outer = optic_to_nt(c.outer), inner = optic_to_nt(c.inner))


"""
    nt_to_optic(nt)

Convert a named tuple representation back to an optic.
"""
function nt_to_optic(nt)
    if nt.type == "identity"
        return identity
    elseif nt.type == "index"
        return IndexLens(eval(Meta.parse(nt.indices)))
    elseif nt.type == "property"
        return PropertyLens{Symbol(nt.field)}()
    elseif nt.type == "composed"
        return nt_to_optic(nt.outer) ∘ nt_to_optic(nt.inner)
    end
end

"""
    vn_to_string(vn::VarName)

Convert a `VarName` as a string, via an intermediate named tuple. This differs
from `string(vn)` in that concretised slices are faithfully represented (rather
than being pretty-printed as colons).

```jldoctest
julia> vn_to_string(@varname(x))
"(sym = \\"x\\", optic = (type = \\"identity\\",))"

julia> vn_to_string(@varname(x.a))
"(sym = \\"x\\", optic = (type = \\"property\\", field = \\"a\\"))"

julia> y = ones(2); vn_to_string(@varname(y[:]))
"(sym = \\"y\\", optic = (type = \\"index\\", indices = \\"(:,)\\"))"

julia> y = ones(2); vn_to_string(@varname(y[:], true))
"(sym = \\"y\\", optic = (type = \\"index\\", indices = \\"(ConcretizedSlice(Base.OneTo(2)),)\\"))"
```
"""
vn_to_string(vn::VarName) = repr((sym = String(getsym(vn)), optic = optic_to_nt(getoptic(vn))))

"""
    vn_from_string(str)

Convert a string representation of a `VarName` back to a `VarName`. The string
should have been generated by `vn_to_string`.

!!! warning
    This function should only be used with trusted input, as it uses `eval`
and `Meta.parse` to parse the string.
"""
function vn_from_string(str)
    new_fields = eval(Meta.parse(str))
    return VarName{Symbol(new_fields.sym)}(nt_to_optic(new_fields.optic))
end

# -----------------------------------------
# Alternate implementation with StructTypes
# -----------------------------------------

index_to_dict(i::Integer) = Dict(:type => "integer", :value => i)
index_to_dict(r::UnitRange) = Dict(:type => "unitrange", :first => first(r), :last => last(r))
index_to_dict(::Colon) = Dict(:type => "colon")
index_to_dict(s::ConcretizedSlice{T,Base.OneTo{I}}) where {T,I} = Dict(:type => "concretized_slice", :oneto => s.range.stop)
index_to_dict(::ConcretizedSlice{T,R}) where {T,R} = error("ConcretizedSlice with range type $(R) not supported")
index_to_dict(t::Tuple) = Dict(:type => "tuple", :values => [index_to_dict(x) for x in t])

function dict_to_index(dict)
    # conversion needed because of the same reason as in dict_to_optic
    dict = Dict(Symbol(k) => v for (k, v) in dict)
    if dict[:type] == "integer"
        return dict[:value]
    elseif dict[:type] == "unitrange"
        return dict[:first]:dict[:last]
    elseif dict[:type] == "colon"
        return Colon()
    elseif dict[:type] == "concretized_slice"
        return ConcretizedSlice(Base.Slice(Base.OneTo(dict[:oneto])))
    elseif dict[:type] == "tuple"
        return tuple(map(dict_to_index, dict[:values])...)
    else
        error("Unknown index type: $(dict[:type])")
    end
end


optic_to_dict(::typeof(identity)) = Dict(:type => "identity")
optic_to_dict(::PropertyLens{sym}) where {sym} = Dict(:type => "property", :field => String(sym))
optic_to_dict(i::IndexLens) = Dict(:type => "index", :indices => index_to_dict(i.indices))
optic_to_dict(c::ComposedOptic) = Dict(:type => "composed", :outer => optic_to_dict(c.outer), :inner => optic_to_dict(c.inner))

function dict_to_optic(dict)
    # Nested dicts are deserialised to Dict{String, Any}
    # but the top level dict is deserialised to Dict{Symbol, Any}
    # so for this recursive function to work we need to first
    # convert String keys to Symbols
    dict = Dict(Symbol(k) => v for (k, v) in dict)
    if dict[:type] == "identity"
        return identity
    elseif dict[:type] == "index"
        return IndexLens(dict_to_index(dict[:indices]))
    elseif dict[:type] == "property"
        return PropertyLens{Symbol(dict[:field])}()
    elseif dict[:type] == "composed"
        return dict_to_optic(dict[:outer]) ∘ dict_to_optic(dict[:inner])
    else
        error("Unknown optic type: $(dict[:type])")
    end
end

struct VarNameWithNTOptic
    sym::Symbol
    optic::Dict{Symbol, Any}
end

function VarNameWithNTOptic(dict::Dict{Symbol, Any})
    return VarNameWithNTOptic{dict[:sym]}(dict[:optic])
end

# Serialisation
StructTypes.StructType(::Type{VarNameWithNTOptic}) = StructTypes.UnorderedStruct()

vn_to_string2(vn::VarName) = JSON3.write(VarNameWithNTOptic(getsym(vn), optic_to_dict(getoptic(vn))))

# Deserialisation
Base.pairs(vn::VarNameWithNTOptic) = Dict(:sym => vn.sym, :optic => vn.optic)

function vn_from_string2(str)
    vn_nt = JSON3.read(str, VarNameWithNTOptic)
    return VarName{vn_nt.sym}(dict_to_optic(vn_nt.optic))
end
