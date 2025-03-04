using Accessors
using Accessors: ComposedOptic, PropertyLens, IndexLens, DynamicIndexLens
using JSON: JSON

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

function subsumes(t::ComposedOptic, u::ComposedOptic)
    return subsumes(t.outer, u.outer) && subsumes(t.inner, u.inner)
end

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
function subsumes(
    t::Union{IndexLens,ComposedOptic{<:ALLOWED_OPTICS,<:IndexLens}},
    u::Union{IndexLens,ComposedOptic{<:ALLOWED_OPTICS,<:IndexLens}},
)
    return subsumes_indices(t, u)
end

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

### Serialisation to JSON / string

# String constants for each index type that we support serialisation /
# deserialisation of
const _BASE_INTEGER_TYPE = "Base.Integer"
const _BASE_VECTOR_TYPE = "Base.Vector"
const _BASE_UNITRANGE_TYPE = "Base.UnitRange"
const _BASE_STEPRANGE_TYPE = "Base.StepRange"
const _BASE_ONETO_TYPE = "Base.OneTo"
const _BASE_COLON_TYPE = "Base.Colon"
const _CONCRETIZED_SLICE_TYPE = "AbstractPPL.ConcretizedSlice"
const _BASE_TUPLE_TYPE = "Base.Tuple"

"""
    index_to_dict(::Integer)
    index_to_dict(::AbstractVector{Int})
    index_to_dict(::UnitRange)
    index_to_dict(::StepRange)
    index_to_dict(::Colon)
    index_to_dict(::ConcretizedSlice{T, Base.OneTo{I}}) where {T, I}
    index_to_dict(::Tuple)

Convert an index `i` to a dictionary representation.
"""
index_to_dict(i::Integer) = Dict("type" => _BASE_INTEGER_TYPE, "value" => i)
index_to_dict(v::Vector{Int}) = Dict("type" => _BASE_VECTOR_TYPE, "values" => v)
function index_to_dict(r::UnitRange)
    return Dict("type" => _BASE_UNITRANGE_TYPE, "start" => r.start, "stop" => r.stop)
end
function index_to_dict(r::StepRange)
    return Dict(
        "type" => _BASE_STEPRANGE_TYPE,
        "start" => r.start,
        "stop" => r.stop,
        "step" => r.step,
    )
end
function index_to_dict(r::Base.OneTo{I}) where {I}
    return Dict("type" => _BASE_ONETO_TYPE, "stop" => r.stop)
end
index_to_dict(::Colon) = Dict("type" => _BASE_COLON_TYPE)
function index_to_dict(s::ConcretizedSlice{T,R}) where {T,R}
    return Dict("type" => _CONCRETIZED_SLICE_TYPE, "range" => index_to_dict(s.range))
end
function index_to_dict(t::Tuple)
    return Dict("type" => _BASE_TUPLE_TYPE, "values" => map(index_to_dict, t))
end

"""
    dict_to_index(dict)
    dict_to_index(symbol_val, dict)

Convert a dictionary representation of an index `dict` to an index.

Users can extend the functionality of `dict_to_index` (and hence `VarName`
de/serialisation) by extending this method along with [`index_to_dict`](@ref).
Specifically, suppose you have a custom index type `MyIndexType` and you want
to be able to de/serialise a `VarName` containing this index type. You should
then implement the following two methods:

1. `AbstractPPL.index_to_dict(i::MyModule.MyIndexType)` should return a
   dictionary representation of the index `i`. This dictionary must contain the
   key `"type"`, and the corresponding value must be a string that uniquely
   identifies the index type. Generally, it makes sense to use the name of the
   type (perhaps prefixed with module qualifiers) as this value to avoid
   clashes. The remainder of the dictionary can have any structure you like.

2. Suppose the value of `index_to_dict(i)["type"]` is `"MyModule.MyIndexType"`.
   You should then implement the corresponding method
   `AbstractPPL.dict_to_index(::Val{Symbol("MyModule.MyIndexType")}, dict)`,
   which should take the dictionary representation as the second argument and
   return the original `MyIndexType` object.

To see an example of this in action, you can look in the the AbstractPPL test
suite, which contains a test for serialising OffsetArrays.
"""
function dict_to_index(dict)
    t = dict["type"]
    if t == _BASE_INTEGER_TYPE
        return dict["value"]
    elseif t == _BASE_VECTOR_TYPE
        return collect(Int, dict["values"])
    elseif t == _BASE_UNITRANGE_TYPE
        return dict["start"]:dict["stop"]
    elseif t == _BASE_STEPRANGE_TYPE
        return dict["start"]:dict["step"]:dict["stop"]
    elseif t == _BASE_ONETO_TYPE
        return Base.OneTo(dict["stop"])
    elseif t == _BASE_COLON_TYPE
        return Colon()
    elseif t == _CONCRETIZED_SLICE_TYPE
        return ConcretizedSlice(Base.Slice(dict_to_index(dict["range"])))
    elseif t == _BASE_TUPLE_TYPE
        return tuple(map(dict_to_index, dict["values"])...)
    else
        # Will error if the method is not defined, but this hook allows users
        # to extend this function
        return dict_to_index(Val(Symbol(t)), dict)
    end
end

optic_to_dict(::typeof(identity)) = Dict("type" => "identity")
function optic_to_dict(::PropertyLens{sym}) where {sym}
    return Dict("type" => "property", "field" => String(sym))
end
optic_to_dict(i::IndexLens) = Dict("type" => "index", "indices" => index_to_dict(i.indices))
function optic_to_dict(c::ComposedOptic)
    return Dict(
        "type" => "composed",
        "outer" => optic_to_dict(c.outer),
        "inner" => optic_to_dict(c.inner),
    )
end

function dict_to_optic(dict)
    if dict["type"] == "identity"
        return identity
    elseif dict["type"] == "index"
        return IndexLens(dict_to_index(dict["indices"]))
    elseif dict["type"] == "property"
        return PropertyLens{Symbol(dict["field"])}()
    elseif dict["type"] == "composed"
        return dict_to_optic(dict["outer"]) ∘ dict_to_optic(dict["inner"])
    else
        error("Unknown optic type: $(dict["type"])")
    end
end

function varname_to_dict(vn::VarName)
    return Dict("sym" => getsym(vn), "optic" => optic_to_dict(getoptic(vn)))
end

function dict_to_varname(dict::Dict{<:AbstractString,Any})
    return VarName{Symbol(dict["sym"])}(dict_to_optic(dict["optic"]))
end

"""
    varname_to_string(vn::VarName)

Convert a `VarName` as a string, via an intermediate dictionary. This differs
from `string(vn)` in that concretised slices are faithfully represented (rather
than being pretty-printed as colons).

For `VarName`s which index into an array, this function will only work if the
indices can be serialised. This is true for all standard Julia index types, but
if you are using custom index types, you will need to implement the
`index_to_dict` and `dict_to_index` methods for those types. See the
documentation of [`dict_to_index`](@ref) for instructions on how to do this.

```jldoctest
julia> varname_to_string(@varname(x))
"{\\"optic\\":{\\"type\\":\\"identity\\"},\\"sym\\":\\"x\\"}"

julia> varname_to_string(@varname(x.a))
"{\\"optic\\":{\\"field\\":\\"a\\",\\"type\\":\\"property\\"},\\"sym\\":\\"x\\"}"

julia> y = ones(2); varname_to_string(@varname(y[:]))
"{\\"optic\\":{\\"indices\\":{\\"values\\":[{\\"type\\":\\"Base.Colon\\"}],\\"type\\":\\"Base.Tuple\\"},\\"type\\":\\"index\\"},\\"sym\\":\\"y\\"}"

julia> y = ones(2); varname_to_string(@varname(y[:], true))
"{\\"optic\\":{\\"indices\\":{\\"values\\":[{\\"range\\":{\\"stop\\":2,\\"type\\":\\"Base.OneTo\\"},\\"type\\":\\"AbstractPPL.ConcretizedSlice\\"}],\\"type\\":\\"Base.Tuple\\"},\\"type\\":\\"index\\"},\\"sym\\":\\"y\\"}"
```
"""
varname_to_string(vn::VarName) = JSON.json(varname_to_dict(vn))

"""
    string_to_varname(str::AbstractString)

Convert a string representation of a `VarName` back to a `VarName`. The string
should have been generated by `varname_to_string`.
"""
string_to_varname(str::AbstractString) = dict_to_varname(JSON.parse(str))

### Prefixing and unprefixing

"""
    _strip_identity(optic)

Remove an inner layer of the identity lens from a composed optic.
"""
_strip_identity(o::Base.ComposedFunction{Outer,typeof(identity)}) where {Outer} = o.outer
_strip_identity(o::Base.ComposedFunction) = o
_strip_identity(o::Accessors.PropertyLens) = o
_strip_identity(o::Accessors.IndexLens) = o
_strip_identity(o::typeof(identity)) = o

"""
    _inner(optic)

Get the innermost (non-identity) layer of an optic.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._inner(Accessors.@o _.a.b.c)
(@o _.a)

julia> AbstractPPL._inner(Accessors.@o _[1][2][3])
(@o _[1])

julia> AbstractPPL._inner(Accessors.@o _)
identity (generic function with 1 method)
```
"""
_inner(o::Base.ComposedFunction{Outer,Inner}) where {Outer,Inner} = o.inner
function _inner(o::Base.ComposedFunction{Outer,typeof(identity)}) where {Outer}
    return _strip_identity(o.outer)
end
_inner(o::Accessors.PropertyLens) = o
_inner(o::Accessors.IndexLens) = o
_inner(o::typeof(identity)) = o

"""
    _outer(optic)

Get the outer layer of an optic.

```jldoctest; setup=:(using Accessors)
julia> AbstractPPL._outer(Accessors.@o _.a.b.c)
(@o _.b.c)

julia> AbstractPPL._outer(Accessors.@o _[1][2][3])
(@o _[2][3])

julia> AbstractPPL._outer(Accessors.@o _.a)
identity (generic function with 1 method)

julia> AbstractPPL._outer(Accessors.@o _[1])
identity (generic function with 1 method)

julia> AbstractPPL._outer(Accessors.@o _)
identity (generic function with 1 method)
```
"""
_outer(o::Base.ComposedFunction{Outer,Inner}) where {Outer,Inner} = _strip_identity(o.outer)
_outer(::Accessors.PropertyLens) = identity
_outer(::Accessors.IndexLens) = identity
_outer(::typeof(identity)) = identity

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
function optic_to_vn(o::Base.ComposedFunction{Outer,typeof(identity)}) where {Outer}
    return optic_to_vn(o.outer)
end
function optic_to_vn(
    o::Base.ComposedFunction{Outer,Accessors.PropertyLens{sym}}
) where {Outer,sym}
    return VarName{sym}(o.outer)
end
function optic_to_vn(@nospecialize(o))
    msg = "optic_to_vn: could not convert optic `$o` to a VarName"
    throw(ArgumentError(msg))
end

unprefix_optic(o, ::typeof(identity)) = o  # Base case
function unprefix_optic(optic, optic_prefix)
    # strip one layer of the optic and check for equality
    inner = _inner(optic)
    inner_prefix = _inner(optic_prefix)
    if inner != inner_prefix
        msg = "could not remove prefix $(optic_prefix) from optic $(optic)"
        throw(ArgumentError(msg))
    end
    # recurse
    return unprefix_optic(_outer(optic), _outer(optic_prefix))
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
    # Special case `identity` to avoid having ComposedFunctions with identity
    if optic_vn == identity
        new_inner_optic_vn = PropertyLens{sym_vn}()
    else
        new_inner_optic_vn = optic_vn ∘ PropertyLens{sym_vn}()
    end
    if optic_prefix == identity
        new_optic_vn = new_inner_optic_vn
    else
        new_optic_vn = new_inner_optic_vn ∘ optic_prefix
    end
    return VarName{sym_prefix}(new_optic_vn)
end
