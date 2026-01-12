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
function Base.isequal(x::VarName, y::VarName)
    return getsym(x) == getsym(y) && isequal(getoptic(x), getoptic(y))
end

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
evaluated. This will convert any `begin` and `end` indices in `vn` to concrete indices with
information about the length of the dimension being indexed into.
"""
concretize(vn::VarName{sym}, x) where {sym} = VarName{sym}(concretize(getoptic(vn), x))

"""
    is_dynamic(vn::VarName)

Return `true` if `vn` contains any dynamic indices (i.e., `begin` and `end`). If a
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
    VarNameConcretizationException()

When constructing a `VarName` using [`@varname`](@ref) (or [`@opticof`](@ref)), we allow
for interpolation of the top-level symbol, e.g. using `name = :x; @varname(\$name)`. However,
if this is done, it is not possible to automatically concretize the resulting `VarName` by
passing `true` as the second argument to `@varname`.

Because macros are confusing, this is probably worth more explanation. For example, consider
the user input `name = :x; @varname(\$name, true)`.

Without concretization, we can easily handle this as `VarName{name}(Iden())`. `name` is then
resolved outside the macro to produce `VarName{:x}(Iden())`. However, to correctly
concretize this, we would need to generate the output `concretize(VarName{name}(), x)`;
i.e., we need to know at macro-expansion time that `name` evaluates to `:x`. This is not
possible given the expression `\$name` alone, which is why this error is thrown.
"""
struct VarNameConcretizationException <: Exception end
function Base.showerror(io::IO, ::VarNameConcretizationException)
    return print(
        io,
        "cannot automatically concretize VarName with interpolated top-level symbol; call `concretize(vn, val)` manually instead",
    )
end

"""
    @varname(expr, concretize=false)

Create a [`VarName`](@ref) given an expression `expr` representing a variable or part of it.

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

Some expressions may involve dynamic indices, specifically, `begin`, `end`. These indices
cannot be resolved, or 'concretized', until the value being indexed into is known. By
default, `@varname(...)` will not automatically concretize these expressions, and thus the
resulting `VarName` will contain markers for these.

Note that colons are not considered dynamic.

```jldoctest
julia> vn = @varname(x[end])
x[DynamicIndex(end)]

julia> vn = @varname(x[1, end-1])
x[1, DynamicIndex(end - 1)]
```

You can detect whether a `VarName` contains any dynamic indices using [`is_dynamic`](@ref):

```jldoctest
julia> vn = @varname(x[1, end-1]); AbstractPPL.is_dynamic(vn)
true
```

To concretize such expressions, you can call [`concretize`](@ref) on the resulting
`VarName`. After concretization, the resulting `VarName` will no longer be dynamic.

```jldoctest
julia> x = randn(2, 3);

julia> vn = @varname(x[1, end-1]); vn2 = AbstractPPL.concretize(vn, x)
x[1, 2]

julia> getoptic(vn2).ix  # Just an ordinary tuple.
(1, 2)

julia> AbstractPPL.is_dynamic(vn2)
false
```

Alternatively, you can pass `true` as the second positional argument to the `@varname` macro
(note that it is not a keyword argument!). This will automatically call [`concretize`](@ref)
for you, using the top-level symbol to look up the value used for concretization.

```jldoctest
julia> x = randn(2, 3);

julia> @varname(x[1:end, end][:], true)
x[1:2, 3][:]
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

For indices, you do not need to use `\$` to interpolate, just use the variable directly:

```jldoctest
julia> ix = 2; @varname(x[ix])
x[2]
```

Note that if the top-level symbol is interpolated, automatic concretization is not possible:

```jldoctest
julia> name = :x; @varname(\$name[1:end], true)
ERROR: LoadError: cannot automatically concretize VarName with interpolated top-level symbol; call `concretize(vn, val)` manually instead
[...]
```
"""
macro varname(expr, concretize::Bool=false)
    return varname(expr, concretize)
end

"""
    varname(expr, concretize::Bool)

Implementation of the `@varname` macro. See the documentation for `@varname` for details.
This function is exported to allow other macros (e.g. in DynamicPPL) to reuse the same
logic.
"""
function varname(expr, concretize::Bool)
    unconcretized_vn, sym = _varname(expr, :($(Iden)()))
    return if concretize
        sym === nothing && throw(VarNameConcretizationException())
        :($(AbstractPPL.concretize)($unconcretized_vn, $(esc(sym))))
    else
        unconcretized_vn
    end
end
function _varname(@nospecialize(expr::Any), ::Any)
    # fallback: it's not a variable!
    throw(VarNameParseException(expr))
end
function _varname(sym::Symbol, inner_expr)
    return :($(VarName){$(QuoteNode(sym))}($inner_expr)), sym
end
function _varname(expr::Expr, inner_expr)
    if Meta.isexpr(expr, :$, 1)
        # Interpolation of the top-level symbol e.g. @varname($name). If we hit this branch,
        # it means that there are no further property/indexing accesses (because otherwise
        # expr.head would be :ref or :.) Thus we don't need to recurse further, and we can
        # just return `inner_expr` as-is.
        sym_expr = expr.args[1]
        return :($(VarName){$(esc(sym_expr))}($inner_expr)), nothing
    else
        next_inner = if expr.head == :(.)
            sym = _handle_property(expr.args[2], expr)
            :($(Property){$(sym)}($inner_expr))
        elseif expr.head == :ref
            original_ixs = expr.args[2:end]
            positional_args = []
            keyword_args = []
            for (dim, ix_expr) in enumerate(original_ixs)
                if _is_kw(ix_expr)
                    push!(keyword_args, :($(ix_expr.args[1]) = $(esc(ix_expr.args[2]))))
                else
                    push!(positional_args, (dim, ix_expr))
                end
            end
            is_single_index = length(positional_args) == 1
            positional_ixs = map(positional_args) do (dim, ix_expr)
                _handle_index(ix_expr, is_single_index ? nothing : dim)
            end
            positional_expr = Expr(:tuple, positional_ixs...)
            kwarg_expr = if isempty(keyword_args)
                :((;))
            else
                Expr(:tuple, keyword_args...)
            end
            :($(Index)($positional_expr, $kwarg_expr, $inner_expr))
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

_is_kw(e::Expr) = Meta.isexpr(e, :kw, 2)
_is_kw(::Any) = false
_handle_index(ix::Any, ::Any) = ix
_handle_index(ix::Symbol, dim) = _make_dynamicindex_expr(ix, dim)
_handle_index(ix::Expr, dim) = _make_dynamicindex_expr(ix, dim)

"""
    @opticof(expr, concretize=false)

Extract the optic from `@varname(expr, concretize)`. This is a thin wrapper around
`getoptic(@varname(...))`.

If you don't need to concretize, you should use `_` as the top-level symbol to
indicate that it is not relevant:

```jldoctest
julia> @opticof(_.a.b)
Optic(.a.b)
```

If you need to concretize, then you can provide a real variable name (which is then used to
look up the value for concretization):

```jldoctest
julia> x = randn(3, 4); @opticof(x[1:end, end], true)
Optic([1:3, 4])
```

Note that concretization with `@opticof` has the same limitations as with `@varname`,
specifically, if the top-level symbol is interpolated, automatic concretization is not
possible.
"""
macro opticof(expr, concretize::Bool=false)
    return :(getoptic($(varname(expr, concretize))))
end

"""
    varname_to_optic(vn::VarName)

Convert a `VarName` to an optic, by converting the top-level symbol to a `Property` optic.
"""
varname_to_optic(vn::VarName{sym}) where {sym} = Property{sym}(getoptic(vn))

"""
    optic_to_varname(optic::Property{sym}) where {sym}

Convert a `Property` optic to a `VarName`, by converting the top-level property to a symbol.
This fails for all other optics.
"""
optic_to_varname(optic::Property{sym}) where {sym} = VarName{sym}(otail(optic))
function optic_to_varname(::AbstractOptic)
    throw(ArgumentError("optic_to_varname: can only convert Property optics to VarName"))
end

"""
    append_optic(vn::VarName, optic::AbstractOptic)

Compose `optic` with the optic in `vn`, returning a new `VarName`.

`optic` is placed at the tail of the existing optic, e.g.

```jldoctest
julia> vn = @varname(x.a.b)
x.a.b

julia> append_optic(vn, @opticof(_[1]))
x.a.b[1]
```
"""
function append_optic(vn::VarName{sym}, optic::AbstractOptic) where {sym}
    return VarName{sym}(cat(getoptic(vn), optic))
end
