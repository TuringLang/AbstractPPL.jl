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
Optic([1][2:3])

julia> getoptic(@varname(y))
Optic()
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
    return _pretty_print_optic(io, getoptic(vn))
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
evaluated. This will convert any `begin`, `end`, or `:` indices in `vn` to concrete indices
with information about the length of the dimension being sliced.
"""
concretize(vn::VarName{sym}, x) where {sym} = VarName{sym}(concretize(getoptic(vn), x))

"""
    is_dynamic(vn::VarName)

Return `true` if `vn` contains any dynamic indices (i.e., `begin`, `end`, or `:`). If a
`VarName` has been concretized, this will always return `false`.
"""
is_dynamic(vn::VarName) = is_dynamic(getoptic(vn))

"""
    VarNameParseException(expr)

An exception thrown when a variable name expression cannot be parsed by the
[`@varname`](@ref) macro.
"""
struct VarNameParseException <: Exception
    expr
end
function Base.showerror(io::IO, e::VarNameParseException)
    return print(io, "malformed variable name `$(e.expr)`")
end

"""
    @varname(expr, concretize=false)

Create a [`VarName`](@ref) given an expression `expr` representing a variable or part of it
(any expression that can be assigned to).

# Basic examples

In general, `VarName`s must have a top-level symbol representing the identifier itself, and
can then have any number of property accesses or indexing operations chained to it.

```jldoctest
julia> @varname(x)
x

julia> @varname(x.a.b.c)
x.a.b.c

julia> @varname(x[1][2][3])
x[1][2][3]

julia> @varname(x.a[1:3].b[2])
x.a[1:3].b[2]
```

# Dynamic indices

Some expressions may involve dynamic indices, e.g., `begin`, `end`, and `:`. These indices
cannot be resolved, or 'concretized', until the value being indexed into is known. By
default, `@varname(...)` will not automatically concretize these expressions, and thus
the resulting `VarName` will contain markers for these.

```jldoctest
julia> # VarNames are pretty-printed, so at first glance, it's not special...
       vn = @varname(x[end])
x[end]

julia> # But if you look under the hood, you can see that the index is dynamic.
       vn = @varname(x[end]); getoptic(vn).ix
(DynamicEnd(),)

julia> vn = @varname(x[1:end, end]); getoptic(vn).ix
(DynamicRange{Int64, DynamicEnd}(1, DynamicEnd()), DynamicEnd())
```

You can detect whether a `VarName` contains any dynamic indices using `is_dynamic(vn)`:

```jldoctest
julia> vn = @varname(x[1:end, end]); is_dynamic(vn)
true
```

To concretize such expressions, you can call `concretize(vn, val)` on the resulting
`VarName`. After concretization, the resulting `VarName` will no longer be dynamic.

```jldoctest
julia> x = randn(2, 3);

julia> vn = @varname(x[1:end, end]); vn2 = concretize(vn, x)
x[1:2, 3][1:2]

julia> getoptic(vn2).ix
((1:2), 3)

julia> is_dynamic(vn2)
false
```

Alternatively, you can pass `true` as the second positional argument the `@varname` macro
(note that it is not a keyword argument!). This will call `concretize` for you, using the
top-level symbol to look up the value used for concretization.

```jldoctest
julia> x = randn(2, 3);

julia> @varname(x[1:end, end][:], true)
x[1:2, 3][1:2]
```

# Interpolation

Property names, as well as top-level symbols, can also be constructed from interpolated
symbols:

```jldoctest
julia> name = :hello; @varname(x.\$name)
x.hello

julia> @varname(\$name)
hello

julia> @varname(\$name.a.\$name[1])
hello.a.hello[1]
```

For indices, you don't need to use `\$` to interpolate, just use the variable directly:

```jldoctest
julia> ix = 2; @varname(x[ix])
x[2]
```
"""
macro varname(expr, concretize::Bool=false)
    unconcretized_vn, sym = _varname(expr, :(Iden()))
    return if concretize
        if sym === nothing
            throw(
                ArgumentError(
                    "cannot automatically concretize VarName with interpolated top-level symbol; call `concretize(vn, val)` manually instead",
                ),
            )
        end
        :(concretize($unconcretized_vn, $(esc(sym))))
    else
        unconcretized_vn
    end
end
function _varname(@nospecialize(expr::Any), ::Any)
    # fallback: it's not a variable!
    throw(VarNameParseException(expr))
end
function _varname(sym::Symbol, inner_expr)
    return :($VarName{$(QuoteNode(sym))}($inner_expr)), sym
end
function _varname(expr::Expr, inner_expr)
    if Meta.isexpr(expr, :$, 1)
        # Interpolation of the top-level symbol e.g. @varname($name). If we hit this branch,
        # it means that there are no further property/indexing accesses (because otherwise
        # expr.head would be :ref or :.) Thus we don't need to recurse further, and we can
        # just return `inner_expr` as-is.
        # TODO(penelopeysm): Is there a way to make auto-concretisation work here? To 
        # be clear, what we want is something like the following to work:
        #    name = :hello; hello = rand(3); @varname($name[:], true)
        # I've tried every combination of `esc`, `QuoteNode`, and `$` I can think of, but
        # with no success yet. It didn't work with old AbstractPPL either ("syntax:
        # all-underscore identifiers are write-only and their values cannot be used in
        # expressions"); at least now we give a more sensible error message.
        sym_expr = expr.args[1]
        return :($VarName{$(sym_expr)}($inner_expr)), nothing
    else
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
