"""
    VarName{sym}(optic=identity)

A variable identifier for a symbol `sym` and optic `optic`. `sym` refers to the name of the 
top-level Julia variable, while `optic` allows one to specify a particular property or index
inside that variable.

`VarName`s can be manually constructed using the `VarName{sym}(optic)` constructor, or from
an optic expression through the [`@varname`](@ref) convenience macro.
"""
struct VarName{sym,T<:AbstractOptic}
    optic::T
    function VarName{sym}(optic=Iden()) where {sym}
        return new{sym,typeof(optic)}(optic)
    end
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
getsym(::VarName{sym}) where {sym} = sym

"""
    getoptic(vn::VarName)

Return the optic of the Julia variable used to generate `vn`.

## Examples

```jldoctest
julia> getoptic(@varname(x[1][2:3]))
[1][2:3]

julia> getoptic(@varname(y))
Iden()
```
"""
getoptic(vn::VarName) = vn.optic

function Base.:(==)(x::VarName, y::VarName)
    return getsym(x) == getsym(y) && getoptic(x) == getoptic(y)
end
Base.isequal(x::VarName, y::VarName) = x == y

Base.hash(vn::VarName, h::UInt) = hash((getsym(vn), getoptic(vn)), h)

function Base.show(io::IO, vn::VarName{sym,T}) where {sym,T}
    print(io, getsym(vn))
    return pretty_print_optic(io, getoptic(vn))
end

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
Base.Symbol(vn::VarName) = Symbol(string(vn))

"""
    concretize(vn::VarName, x)

Return `vn` concretized on `x`, i.e. any information related to the runtime shape of `x` is
evaluated. This will convert any Colon indices to `Base.Slice`, which contains information
about the length of the dimension being sliced.
"""
# TODO(penelopeysm): Does this affect begin/end? The old docstring said it would, but I
# could not see where in the implementation this was actually done. I remember that this is
# not the first time I've been confused about this.
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

# Interpolation

Property names can also be constructed from interpolated symbols:

```jldoctest
julia> name = :hello; @varname(x.\$name)
x.hello
```

For indices, you don't need to use `\$` to interpolate, just use the variable directly:

```jldoctest
julia> ix = 2; @varname(x[ix])
x[2]
```
"""

struct VarNameParseException <: Exception
    expr::Expr
end
function Base.showerror(io::IO, e::VarNameParseException)
    return print(io, "malformed variable name `$(e.expr)`")
end

macro varname(expr)
    return _varname(expr, :(Iden()))
end
function _varname(sym::Symbol, inner_expr)
    return :($VarName{$(QuoteNode(sym))}($inner_expr))
end
function _varname(expr::Expr, inner_expr)
    next_inner = if expr.head == :(.)
        sym = _handle_property(expr.args[2], expr)
        :(Property{$(sym)}($inner_expr))
    elseif expr.head == :ref
        ixs = map(first ∘ _handle_index, expr.args[2:end])
        # TODO(penelopeysm): Technically, here we could track whether any of the indices are
        # dynamic, and store this for later use.
        #     isdyn = any(last, ixs_and_isdyn)
        # What we do now (generate the dynamic VarName first, and then later check whether
        # it needs concretization) is slightly inefficient.
        :(Index(tuple($(ixs...)), $inner_expr))
    else
        # some other expression we can't parse
        throw(VarNameParseException(expr))
    end
    return _varname(expr.args[1], next_inner)
end

function _handle_property(qn::QuoteNode, original_expr)
    if qn.value isa Symbol # no interpolation e.g. @varname(x.a)
        return qn
    elseif Meta.isexpr(qn.value, :$, 1) && qn.value.args[1] isa Symbol
        # interpolated property e.g. @varname(x.$name).
        # TODO(penelopeysm): Note that $name must evaluate to a Symbol, or else you will get
        # a slightly inscrutable error: "ERROR: TypeError: in Type, in parameter, expected
        # Type, got a value of type String". This should probably be fixed, but I don't
        # actually *know* how to do it. Again, this is not a new issue, the old VarName
        # also had the same problem.
        return esc(qn.value.args[1])
    else
        throw(VarNameParseException(original_expr))
    end
end
function _handle_property(::Any, original_expr)
    throw(VarNameParseException(original_expr))
end

_handle_index(ix::Int) = ix, false
function _handle_index(ix::Symbol)
    # NOTE(penelopeysm): We could just use `:end` instead of Symbol(:end), but the former
    # messes up syntax highlighting with Treesitter
    # https://github.com/tree-sitter/tree-sitter-julia/issues/104
    if ix == Symbol(:end)
        return :(DynamicEnd()), true
    elseif ix == Symbol(:begin)
        return :(DynamicBegin()), true
    elseif ix == :(:)
        return :(DynamicColon()), true
    else
        # an interpolated symbol
        return ix, false
    end
end
function _handle_index(ix::Expr)
    if Meta.isexpr(ix, :call, 3) && ix.args[1] == :(:)
        # This is a range
        start, isdyn = _handle_index(ix.args[2])
        stop, isdyn2 = _handle_index(ix.args[3])
        if isdyn || isdyn2
            return :(DynamicRange($start, $stop)), true
        else
            return :(($start):($stop)), false
        end
    else
        # Some other expression. We don't want to parse this any further, but we also don't
        # want to error, because it may well be an expression that evaluates to a valid
        # index.
        return ix, false
    end
end

#=
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

=#
