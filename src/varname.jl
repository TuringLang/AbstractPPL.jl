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

```jldoctest
julia> vn = VarName{:x}(((Colon(), 1), (2,)))
x[:,1][2]

julia> vn.indexing
((Colon(), 1), (2,))

julia> @varname x[:, 1][1+1]
x[:,1][2]
```
"""
struct VarName{sym, T<:Tuple}
    indexing::T

    VarName{sym}(indexing::Tuple=()) where {sym} = new{sym,typeof(indexing)}(indexing)
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
function VarName(vn::VarName, indexing::Tuple = ())
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
((1,), (2:3,))

julia> getindexing(@varname(y))
()
```
"""
getindexing(vn::VarName) = vn.indexing


Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getindexing(vn)), h)
Base.:(==)(x::VarName, y::VarName) = getsym(x) == getsym(y) && getindexing(x) == getindexing(y)

function Base.show(io::IO, vn::VarName)
    print(io, getsym(vn))
    for indices in getindexing(vn)
        print(io, "[")
        join(io, map(prettify_index, indices), ",")
        print(io, "]")
    end
end

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

subsumes(::Tuple{}, ::Tuple{}) = true  # x subsumes x
subsumes(::Tuple{}, ::Tuple) = true    # x subsumes x[1]
subsumes(::Tuple, ::Tuple{}) = false   # x[1] does not subsume x
function subsumes(t::Tuple, u::Tuple)  # does x[i]... subsume x[j]...?
    return _issubindex(first(t), first(u)) && subsumes(Base.tail(t), Base.tail(u))
end

const AnyIndex = Union{Int, AbstractVector{Int}, Colon} 
_issubindex_(::Tuple{Vararg{AnyIndex}}, ::Tuple{Vararg{AnyIndex}}) = false
function _issubindex(t::NTuple{N, AnyIndex}, u::NTuple{N, AnyIndex}) where {N}
    return all(_issubrange(j, i) for (i, j) in zip(t, u))
end

const ConcreteIndex = Union{Int, AbstractVector{Int}} # this include all kinds of ranges

"""Determine whether indices `i` are contained in `j`, treating `:` as universal set."""
_issubrange(i::ConcreteIndex, j::ConcreteIndex) = issubset(i, j)
_issubrange(i::Union{ConcreteIndex, Colon}, j::Colon) = true
_issubrange(i::Colon, j::ConcreteIndex) = true



"""
    @varname(expr)

A macro that returns an instance of [`VarName`](@ref) given a symbol or indexing expression `expr`.

The `sym` value is taken from the actual variable name, and the index values are put appropriately
into the constructor (and resolved at runtime).

NB: `begin` and `end` indexing can be used, but remember that they depend on _runtime values_ --
their usage requires that the array over which the indexing expression is defined is defined, in
order for `firstindex` and `lastindex` to work in the expanded code.

## Examples

```jldoctest
julia> @varname(x).indexing
()

julia> @varname(x[1]).indexing
((1,),)

julia> @varname(x[:, 1]).indexing
((Colon(), 1),)

julia> @varname(x[:, 1][2]).indexing
((Colon(), 1), (2,))

julia> @varname(x[1,2][1+5][45][3]).indexing
((1, 2), (6,), (45,), (3,))

julia> let a = [42]; @varname(a[1][end][3]); end
a[1][1][3]
```

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro varname(expr::Union{Expr, Symbol})
    return esc(varname(expr))
end

varname(sym::Symbol) = :($(AbstractPPL.VarName){$(QuoteNode(sym))}())
function varname(expr::Expr)
    if Meta.isexpr(expr, :ref)
        head = vsym(expr)
        inds = vinds(expr, head)
        return :($(AbstractPPL.VarName){$(QuoteNode(head))}($inds))
    else
        error("Malformed variable name $(expr)!")
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
    if Meta.isexpr(expr, :ref)
        return vsym(expr.args[1])
    else
        error("Malformed variable name $(expr)!")
    end
end

"""
    @vinds(expr)

Returns a tuple of tuples of the indices in `expr`.

## Examples

```jldoctest
julia> @vinds x
()

julia> @vinds x[1,1][2,3]
((1, 1), (2, 3))

julia> @vinds x[:,1][2,:]
((Colon(), 1), (2, Colon()))

julia> @vinds x[2:3,1][2,1:2]
((2:3, 1), (2, 1:2))

julia> @vinds x[2:3,2:3][[1,2],[1,2]]
((2:3, 2:3), ([1, 2], [1, 2]))
```

!!! compat "Julia 1.5"
    Using `begin` in an indexing expression to refer to the first index requires at least
    Julia 1.5.
"""
macro vinds(expr::Union{Expr, Symbol})
    return esc(vinds(expr))
end


@static if VERSION < v"1.5.0-DEV.666"
    _replace_ref_begin_end(ex, withex) = Base.replace_ref_end_!(copy(ex), withex)
    _replace_ref_begin_end(ex::Symbol, withex) = Base.replace_ref_end_!(ex, withex)
    _index_replacement_for(s) = :($lastindex($s))
    _index_replacement_for(s, n) = :($lastindex($s, $n))
else
    _replace_ref_begin_end(ex, withex) = Base.replace_ref_begin_end_!(copy(ex), withex)
    _replace_ref_begin_end(ex::Symbol, withex) = Base.replace_ref_begin_end_!(ex, withex)
    _index_replacement_for(s) = :($firstindex($s)), :($lastindex($s))
    _index_replacement_for(s, n) = :($firstindex($s, $n)), :($lastindex($s, $n))
end


"""
    vinds(expr)

Return the indexing part of the [`@varname`](@ref)-compatible expression `expr` as an expression
suitable for input of the [`VarName`](@ref) constructor (i.e., a tuple of tuples).

## Examples

```jldoctest
julia> x = [10, 20, 30]; eval(vinds(:(x[end])))
((3,),)


julia> x = [10 20]; eval(vinds(:(x[1, end])))
((1, 2),)

julia> x = [[1, 2]]; eval(vinds(:(x[1][end])))
((1,), (2,))

julia> x = ([1, 2], ); eval(vinds(:(x[1][end]))) # tuple
((1,), (2,))

julia> x = [fill([[10], [20, 30]], 2, 2, 2)]
       if VERSION < v"1.5.0-DEV.666"
            eval(vinds(:(x[1][2, end, :][2][end])))
       else
            eval(vinds(Meta.parse("x[begin][2, end, :][2][end]")))
       end
((1,), (2, 2, Colon()), (2,), (2,))

```
"""
function vinds(expr, head = vsym(expr))
    # see https://github.com/JuliaLang/julia/blob/bb5b98e72a151c41471d8cc14cacb495d647fb7f/base/views.jl#L17-L75
    indexing = _straighten_indexing(expr)
    inds = Expr[]  # collection of result indices
    partial = head  # partial :ref expressions, used in caching
    cached_exprs = Vector{Pair{Symbol, Expr}}()  # cache for partial expressions going into a let
    
    for ixs in indexing
        # S becomes the name of the cached variable
        S = (partial == head) ? head : gensym(:S)
        used_S = false
        
        nixs = length(ixs)
        if nixs == 1
            # for 1D indexing, just use `lastindex(x)`
            ixs[1], used = _replace_ref_begin_end(ixs[1], _index_replacement_for(S))
            used_S |= used
        elseif nixs > 1
            # otherwise, we need `lastindex(x, i)`
            for i in eachindex(ixs)
                ixs[i], used = _replace_ref_begin_end(ixs[i], _index_replacement_for(S, i))
                used_S |= used
            end
        end

        if used_S && partial !== head
            # cache that expression if we actually used it, and use the new name in the
            # partial expression
            push!(cached_exprs, S => partial)
            partial = Expr(:call, Base.maybeview, S, ixs...)
        else
            partial = Expr(:call, Base.maybeview, partial, ixs...)
        end
        
        push!(inds, Expr(:tuple, ixs...))
    end

    # finally make the tuple of tuples
    tuple_expr = Expr(:tuple, inds...)
    
    if length(cached_exprs) == 0 
        return tuple_expr
    else
        # construct one big let expression
        cached_assignments = [:($S = $partial) for (S, partial) in cached_exprs]
        return Expr(:let, Expr(:block, cached_assignments...), tuple_expr)
    end
end


"""
    _straighten_indexing(expr)

Extract a list of lists of (raw) indices of an iterated `:ref` expression.

```julia
julia> _straighten_indexing(:(x[begin][2, end, :][2][end]))
4-element Array{Array{Any,1},1}:
 [:begin]
 [2, :end, :(:)]
 [2]
 [:end]
```
"""
_straighten_indexing(expr::Symbol) = Vector{Any}[]
function _straighten_indexing(expr::Expr)
    if Meta.isexpr(expr, :ref)
        init = _straighten_indexing(expr.args[1])
        last = expr.args[2:end]
        return push!(init, last)
    else
        error("Mis-formed variable name $(expr)!")
    end
end






