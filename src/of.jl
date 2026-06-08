using Random: AbstractRNG, default_rng, randexp

"""
    OfType

Abstract base type for all types in the `of` type system.

The `of` type system provides a declarative way to specify parameter types for 
probabilistic programming. All `of` types encode their specifications (dimensions, 
bounds, etc.) in type parameters, allowing them to be used as actual Julia types 
in type annotations.

# Subtypes
- `OfReal{T,Lower,Upper}`: Bounded or unbounded floating-point numbers
- `OfInt{Lower,Upper}`: Bounded or unbounded integers  
- `OfArray{T,N,Dims}`: Arrays with specified element type and dimensions
- `OfNamedTuple{Names,Types}`: Named tuples with typed fields
- `OfConstantWrapper{T}`: Wrapper marking a type as constant/hyperparameter

# See also
[`of`](@ref), [`@of`](@ref)
"""
abstract type OfType end

"""
    SymbolicRef{S}

Wrapper type for symbolic references in bounds and dimensions.

Used internally to encode references to other fields when specifying bounds or dimensions 
that depend on constant fields or other parameters. For example, when using `@of(n=of(Int; constant=true), 
data=of(Array, n, 2))`, the reference to `n` in the array dimension is encoded as 
`SymbolicRef{:n}`.

# Type Parameters
- `S`: The symbol being referenced

# See also
[`@of`](@ref), [`of`](@ref)
"""
struct SymbolicRef{S} end

"""
    SymbolicExpr{E}

Wrapper type for symbolic expressions in dimensions.

Used internally to encode arithmetic expressions involving constant fields or other parameters. For example,
when using `@of(n=of(Int; constant=true), padded=of(Array, n+1, n+1))`, the expression 
`n+1` is encoded as `SymbolicExpr{(:+, :n, 1)}`.

Supported operations: `+`, `-`, `*`, `/`. Division operations must result in integers 
when used for array dimensions.

# Type Parameters
- `E`: A tuple representing the expression in prefix notation

# See also
[`@of`](@ref), [`of`](@ref)
"""
struct SymbolicExpr{E} end

"""
    OfReal{T<:AbstractFloat,Lower,Upper}

Type specification for bounded or unbounded floating-point numbers.

# Type Parameters
- `T<:AbstractFloat`: The concrete floating-point type (e.g., `Float64`, `Float32`)
- `Lower`: Lower bound (numeric value, `Nothing` for unbounded, or `SymbolicRef`)
- `Upper`: Upper bound (numeric value, `Nothing` for unbounded, or `SymbolicRef`)

# Examples
```julia
of(Float64)              # OfReal{Float64, Nothing, Nothing}
of(Float32, 0.0, 1.0)    # OfReal{Float32, 0.0, 1.0}
of(Real, 0, nothing)     # OfReal{Float64, 0, Nothing} (defaults to Float64)
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
struct OfReal{T<:AbstractFloat,Lower,Upper} <: OfType
    function OfReal{T,L,U}() where {T<:AbstractFloat,L,U}
        return error(
            "OfReal is a type specification, not an instantiable object. Use of(Float64, ...) or of(Float32, ...) to create the type.",
        )
    end
end

"""
    OfInt{Lower,Upper}

Type specification for bounded or unbounded integers.

# Type Parameters
- `Lower`: Lower bound (integer value, `Nothing` for unbounded, or `SymbolicRef`)
- `Upper`: Upper bound (integer value, `Nothing` for unbounded, or `SymbolicRef`)

# Examples
```julia
of(Int)           # OfInt{Nothing, Nothing}
of(Int, 1, 10)    # OfInt{1, 10}
of(Int, 0, nothing)  # OfInt{0, Nothing}
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
struct OfInt{Lower,Upper} <: OfType
    function OfInt{L,U}() where {L,U}
        return error(
            "OfInt is a type specification, not an instantiable object. Use of(Int, ...) to create the type.",
        )
    end
end

"""
    OfArray{T,N,Dims}

Type specification for arrays with fixed element type and dimensions.

# Type Parameters
- `T`: Element type of the array
- `N`: Number of dimensions
- `Dims`: Tuple type encoding the size of each dimension (can include `SymbolicRef` or `SymbolicExpr`)

# Examples
```julia
of(Array, 3, 4)          # OfArray{Float64, 2, (3, 4)}
of(Array, Float32, 10)   # OfArray{Float32, 1, (10,)}
@of(n=of(Int; constant=true), data=of(Array, n, 2))  # Symbolic dimension
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
struct OfArray{T,N,Dims} <: OfType
    function OfArray{T,N,D}() where {T,N,D}
        return error(
            "OfArray is a type specification, not an instantiable object. Use of(Array, ...) to create the type.",
        )
    end
end

"""
    OfNamedTuple{Names,Types<:Tuple}

Type specification for named tuples with typed fields.

# Type Parameters
- `Names`: Tuple of field names as symbols
- `Types<:Tuple`: Tuple of field types (each must be an `OfType`)

# Examples
```julia
@of(mu=of(Real), tau=of(Real, 0, nothing))
of((a=of(Int), b=of(Array, 3, 3)))
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
struct OfNamedTuple{Names,Types<:Tuple} <: OfType
    function OfNamedTuple{Names,Types}() where {Names,Types}
        return error(
            "OfNamedTuple is a type specification, not an instantiable object. Use of(...) to create the type.",
        )
    end
end

"""
    OfConstantWrapper{T<:OfType}

Wrapper type marking a field as a constant/hyperparameter.

Constants are not included in flattened representations and must be provided
when creating instances or concretizing types with symbolic dimensions.

# Type Parameters
- `T<:OfType`: The wrapped type specification

# Examples
```julia
of(Int; constant=true)    # OfConstantWrapper{OfInt{Nothing, Nothing}}
of(Real; constant=true)   # OfConstantWrapper{OfReal{Float64, Nothing, Nothing}}
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
struct OfConstantWrapper{T<:OfType} <: OfType
    function OfConstantWrapper{T}() where {T<:OfType}
        return error(
            "OfConstantWrapper is a type specification, not an instantiable object. Use of(...; constant=true) to create the type.",
        )
    end
end

get_lower(::Type{OfReal{T,L,U}}) where {T,L,U} = L
get_upper(::Type{OfReal{T,L,U}}) where {T,L,U} = U
get_element_type(::Type{OfReal{T,L,U}}) where {T,L,U} = T
get_lower(::Type{OfInt{L,U}}) where {L,U} = L
get_upper(::Type{OfInt{L,U}}) where {L,U} = U
get_element_type(::Type{OfArray{T,N,D}}) where {T,N,D} = T
get_ndims(::Type{OfArray{T,N,D}}) where {T,N,D} = N
function get_dims(::Type{OfArray{T,N,D}}) where {T,N,D}
    return D isa DataType && D <: Tuple ? tuple(D.parameters...) : D
end
get_names(::Type{OfNamedTuple{Names,Types}}) where {Names,Types} = Names
get_types(::Type{OfNamedTuple{Names,Types}}) where {Names,Types} = Types
get_wrapped_type(::Type{OfConstantWrapper{T}}) where {T} = T

# A dimension is symbolic when it is a bare field symbol or a `SymbolicExpr` parameter.
_is_symbolic_dim(d) = d isa Symbol || (d isa Type && d <: SymbolicExpr)
# A bound is symbolic when it is a `SymbolicRef` or `SymbolicExpr` parameter.
_is_symbolic_bound(b) = b isa Type && (b <: SymbolicRef || b <: SymbolicExpr)

# Fail fast (with a clear message) before a symbolic bound reaches arithmetic/comparison.
function _assert_concrete_bounds(::Type{T}) where {T<:OfType}
    if _is_symbolic_bound(get_lower(T)) || _is_symbolic_bound(get_upper(T))
        error(
            "Cannot instantiate a type with symbolic bounds. Resolve them with of(T; name=value) first.",
        )
    end
    return nothing
end

is_leaf(::Type{<:OfArray}) = true
is_leaf(::Type{<:OfReal}) = true
is_leaf(::Type{<:OfInt}) = true
is_leaf(::Type{<:OfNamedTuple}) = false
is_leaf(::Type{<:OfConstantWrapper}) = true

bound_to_type(::Nothing) = Nothing
bound_to_type(x::Real) = x
bound_to_type(s::Symbol) = SymbolicRef{s}
bound_to_type(s::QuoteNode) = SymbolicRef{s.value}

type_to_bound(::Type{Nothing}) = nothing
type_to_bound(::Type{x}) where {x<:Real} = x
type_to_bound(::Type{SymbolicRef{S}}) where {S} = S
type_to_bound(s::Symbol) = s
type_to_bound(x::Real) = x

function eval_symbolic_expr(expr::Tuple, bindings::NamedTuple)
    if length(expr) < 2
        error("Invalid expression format: $expr")
    end

    op = expr[1]
    if !(op in (:+, :-, :*, :/))
        error("Unsupported operation: $op. Only +, -, *, / are supported.")
    end

    args = map(expr[2:end]) do arg
        if arg isa Symbol
            if haskey(bindings, arg)
                bindings[arg]
            else
                error("Symbol '$arg' not found in bindings")
            end
        elseif arg isa Tuple
            eval_symbolic_expr(arg, bindings)
        else
            arg
        end
    end

    if op == :+
        return sum(args)
    elseif op == :-
        return length(args) == 1 ? -args[1] : args[1] - args[2]
    elseif op == :*
        return prod(args)
    elseif op == :/
        length(args) == 2 || error("Division requires exactly 2 arguments")
        result = args[1] / args[2]
        # Array dimensions must be integers; guard the result itself, not just the dividend.
        isinteger(result) || error(
            "Division $(args[1]) / $(args[2]) = $result is not an integer. Array dimensions must be integers.",
        )
        return Int(result)
    end
end

# Resolve bound references during type concretization
function resolve_bound(::Type{Nothing}, replacements::NamedTuple)
    return Nothing
end

function resolve_bound(::Type{x}, replacements::NamedTuple) where {x<:Real}
    return x
end

function resolve_bound(::Type{SymbolicRef{S}}, replacements::NamedTuple) where {S}
    if haskey(replacements, S)
        return bound_to_type(replacements[S])
    else
        return SymbolicRef{S}
    end
end

function resolve_bound(::Type{SymbolicExpr{E}}, replacements::NamedTuple) where {E}
    evaluated = eval_symbolic_expr(E, replacements)
    return bound_to_type(evaluated)
end

function resolve_bound(T::Type, ::NamedTuple)
    return T
end

function resolve_bound(x::Real, ::NamedTuple)
    return x
end

function process_array_dimensions(dims)
    # Unwrap quoted field references; keep everything else (including SymbolicExpr{...}) as-is.
    processed_dims = map(d -> d isa QuoteNode ? d.value : d, dims)
    if length(processed_dims) == 1
        Tuple{processed_dims[1]}
    else
        Tuple{processed_dims...}
    end
end

function process_bounds(lower, upper)
    L = if lower isa Type
        lower
    else
        bound_to_type(lower)
    end
    U = if upper isa Type
        upper
    else
        bound_to_type(upper)
    end
    return L, U
end

"""
    of(T, args...; constant::Bool=false)

Create an `OfType` specification from various inputs.

# Main Methods

## Arrays
```julia
of(Array, dims...)              # Float64 array with given dimensions
of(Array, T, dims...)           # Array with element type T and given dimensions
```

## Real Numbers
```julia
of(Float64)                     # Unbounded Float64
of(Float64, lower, upper)       # Bounded Float64
of(Float32)                     # Unbounded Float32
of(Float32, lower, upper)       # Bounded Float32
of(Real)                        # Unbounded Real (defaults to Float64)
of(Real, lower, upper)          # Bounded Real (defaults to Float64)
```

## Integers
```julia
of(Int)                         # Unbounded integer
of(Int, lower, upper)           # Bounded integer
```

## Named Tuples
```julia
of((;field1=spec1, field2=spec2, ...))  # NamedTuple with typed fields
```

## From Values (Type Inference)
```julia
of(1.0)                         # Infers of(Float64)
of([1, 2, 3])                   # Infers of(Array, Int, 3)
of((a=1, b=2.0))               # Infers OfNamedTuple
```

# Arguments
- `T`: Type to create specification for
- `args...`: Type-specific arguments (bounds, dimensions, etc.)
- `constant`: Mark type as constant/hyperparameter (default: false)

# Returns
An `OfType` subtype encoding the specification in its type parameters.

# Examples
```julia
# Basic types
T1 = of(Float64, 0, 1)          # OfReal{Float64, 0, 1}
T2 = of(Array, 3, 4)            # OfArray{Float64, 2, (3, 4)}
T3 = of(Int; constant=true)     # OfConstantWrapper{OfInt{Nothing, Nothing}}

# With @of macro for cleaner syntax
T4 = @of(
    n = of(Int; constant=true),
    data = of(Array, n, 2)      # Symbolic dimension
)

# Type concretization
T5 = of(T4; n=10)               # Concrete type with n=10
```

# See also
[`@of`](@ref), [`OfType`](@ref)
"""
function of(::Type{Array}, dims...; constant::Bool=false)
    if constant
        error("constant=true is only supported for Int and Real types, not Array")
    end
    # Default to Float64 for unspecified array types
    dims_tuple = process_array_dimensions(dims)
    return OfArray{Float64,length(dims),dims_tuple}
end

function of(::Type{Array}, T::Type, dims...; constant::Bool=false)
    # Check if T is a symbolic expression type (which should be treated as a dimension)
    if T <: SymbolicExpr
        # This is actually a dimension, not an element type
        # Construct the array type directly with Float64 as element type
        if constant
            error("constant=true is only supported for Int and Real types, not Array")
        end
        all_dims = (T, dims...)
        dims_tuple = process_array_dimensions(all_dims)
        return OfArray{Float64,length(all_dims),dims_tuple}
    end

    if constant
        error("constant=true is only supported for Int and Real types, not Array")
    end
    dims_tuple = process_array_dimensions(dims)
    return OfArray{T,length(dims),dims_tuple}
end

function of(::Type{Int}; constant::Bool=false)
    base_type = OfInt{Nothing,Nothing}
    return constant ? OfConstantWrapper{base_type} : base_type
end

function of(
    ::Type{Int},
    lower::Union{Int,Nothing,Symbol,Type},
    upper::Union{Int,Nothing,Symbol,Type};
    constant::Bool=false,
)
    L, U = process_bounds(lower, upper)
    base_type = OfInt{L,U}
    return constant ? OfConstantWrapper{base_type} : base_type
end

function of(::Type{T}; constant::Bool=false) where {T<:AbstractFloat}
    base_type = OfReal{T,Nothing,Nothing}
    return constant ? OfConstantWrapper{base_type} : base_type
end

function of(
    ::Type{T},
    lower::Union{Real,Nothing,Symbol,Type},
    upper::Union{Real,Nothing,Symbol,Type};
    constant::Bool=false,
) where {T<:AbstractFloat}
    L, U = process_bounds(lower, upper)
    base_type = OfReal{T,L,U}
    return constant ? OfConstantWrapper{base_type} : base_type
end

# of(Real) creates Float64
function of(::Type{Real}; constant::Bool=false)
    base_type = OfReal{Float64,Nothing,Nothing}
    return constant ? OfConstantWrapper{base_type} : base_type
end

function of(
    ::Type{Real},
    lower::Union{Real,Nothing,Symbol,Type},
    upper::Union{Real,Nothing,Symbol,Type};
    constant::Bool=false,
)
    L, U = process_bounds(lower, upper)
    base_type = OfReal{Float64,L,U}
    return constant ? OfConstantWrapper{base_type} : base_type
end

# Infer OfType from concrete values
function of(value::T) where {T<:AbstractFloat}
    return of(T)
end

function of(value::Integer)
    return of(Int)
end

# Fallback for other Real types
function of(value::Real)
    return of(Float64)
end

function of(value::AbstractArray{T,N}) where {T,N}
    return of(Array, T, size(value)...)
end

function of(value::NamedTuple{names}) where {names}
    # Check if all values are already OfType types
    vals = values(value)
    if all(v -> v isa Type && v <: OfType, vals)
        # This is a NamedTuple of types, not values
        return OfNamedTuple{names,Tuple{vals...}}
    else
        # This is a NamedTuple of values, infer types
        of_types = map(of, vals)
        return OfNamedTuple{names,Tuple{of_types...}}
    end
end

function resolve_bounded_type(::Type{T}, replacements::NamedTuple) where {T<:OfType}
    if !(T <: OfReal || T <: OfInt)
        return T
    end

    lower = get_lower(T)
    upper = get_upper(T)
    new_lower = resolve_bound(lower, replacements)
    new_upper = resolve_bound(upper, replacements)

    if new_lower !== lower || new_upper !== upper
        if T <: OfReal
            elem_type = get_element_type(T)
            return OfReal{elem_type,new_lower,new_upper}
        elseif T <: OfInt
            return OfInt{new_lower,new_upper}
        end
    else
        return T
    end
end

"""
    of(::Type{T}, replacements::NamedTuple) where T<:OfType
    of(::Type{T}; kwargs...) where T<:OfType
    of(::Type{T}, pairs::Pair{Symbol}...) where T<:OfType

Create a concrete type by resolving symbolic dimensions and removing constants.

This function takes an `OfType` with symbolic dimensions or constants and creates
a new type with some or all symbols resolved to concrete values. Constants that
are provided are removed from the resulting type.

# Arguments
- `T<:OfType`: The type to concretize
- `replacements`: Named tuple or keyword arguments mapping symbols to values

# Returns
A new `OfType` with symbols replaced and constants removed.

# Examples
```julia
# Define type with symbolic dimensions
T = @of(
    n = of(Int; constant=true),
    data = of(Array, n, 2)
)

# Create concrete type
ConcreteT = of(T; n=10)      # @of(data=of(Array, 10, 2))

# Partial concretization
T2 = @of(
    rows = of(Int; constant=true),
    cols = of(Int; constant=true),
    matrix = of(Array, rows, cols)
)
Partial = of(T2; rows=5)     # @of(cols=of(Int; constant=true), matrix=of(Array, 5, :cols))
```

# See also
[`of`](@ref), [`@of`](@ref)
"""
function of(::Type{T}, pairs::Pair{Symbol}...) where {T<:OfType}
    return of(T, NamedTuple(pairs))
end

function of(::Type{T}; kwargs...) where {T<:OfType}
    return of(T, NamedTuple(kwargs))
end

function _resolve_dimensions(dims, replacements::NamedTuple)
    return map(dims) do d
        if d isa Symbol && haskey(replacements, d)
            replacements[d]
        elseif d isa Type && d <: SymbolicExpr
            # Evaluate the expression
            expr = d.parameters[1]
            eval_symbolic_expr(expr, replacements)
        else
            d
        end
    end
end

function _process_field_for_concretization(
    name::Symbol, field_type::Type, replacements::NamedTuple
)
    # Case 1: Constant field that has been resolved - skip it
    if field_type <: OfConstantWrapper && haskey(replacements, name)
        return nothing
    end

    # Case 2: Array with potentially symbolic dimensions
    if field_type <: OfArray
        dims = get_dims(field_type)
        new_dims = _resolve_dimensions(dims, replacements)

        if new_dims != dims
            T = get_element_type(field_type)
            return (name, of(Array, T, new_dims...))
        else
            return (name, field_type)
        end
    end

    # Case 3: Nested NamedTuple - recursively concretize
    if field_type <: OfNamedTuple
        return (name, of(field_type, replacements))
    end

    # Case 4: Bounded types (Real/Int) with potentially symbolic bounds
    if field_type <: OfReal || field_type <: OfInt
        return (name, resolve_bounded_type(field_type, replacements))
    end

    # Case 5: Constant wrapper without replacement - resolve wrapped type
    if field_type <: OfConstantWrapper
        wrapped = get_wrapped_type(field_type)
        resolved = resolve_bounded_type(wrapped, replacements)

        if resolved !== wrapped
            return (name, OfConstantWrapper{resolved})
        else
            return (name, field_type)
        end
    end

    # Case 6: Other types - pass through unchanged
    return (name, field_type)
end

function of(::Type{OfNamedTuple{Names,Types}}, replacements::NamedTuple) where {Names,Types}
    processed_fields = []

    for i in 1:length(Names)
        name = Names[i]
        field_type = Types.parameters[i]

        result = _process_field_for_concretization(name, field_type, replacements)

        if !isnothing(result)
            push!(processed_fields, result)
        end
    end

    # Check if any fields remain
    if isempty(processed_fields)
        error("All fields were constants and have been resolved. No fields remain.")
    end

    # Extract names and types from processed fields
    remaining_names = [field[1] for field in processed_fields]
    remaining_types = [field[2] for field in processed_fields]

    return OfNamedTuple{Tuple(remaining_names),Tuple{remaining_types...}}
end

function of(::Type{OfArray{T,N,D}}, replacements::NamedTuple) where {T,N,D}
    # Replace symbolic dimensions in array types
    dims = get_dims(OfArray{T,N,D})
    new_dims = _resolve_dimensions(dims, replacements)
    return OfArray{T,N,Tuple{new_dims...}}
end

function of(::Type{T}, replacements::NamedTuple) where {T<:OfType}
    # For other types (OfReal, OfConstantWrapper), just return as-is
    return T
end

function _create_with_default(::Type{OfArray{T,N,D}}, default_value) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    if any(d -> d isa Symbol || (d isa Type && d <: SymbolicExpr), dims)
        error(
            "Cannot create array with symbolic dimensions. Use T(default_value; kwargs...) with dimension values.",
        )
    end
    # Handle missing values specially - create array of Union{T,Missing}
    if default_value === missing
        return fill(missing, dims...)
    else
        return fill(convert(T, default_value), dims...)
    end
end

function _create_with_default(::Type{OfReal{T,L,U}}, default_value) where {T,L,U}
    if default_value === missing
        return missing
    end
    val = convert(T, default_value)
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    validate_bounds(val, lower, upper; kind="Real value")
    return val
end

function _create_with_default(::Type{OfInt{L,U}}, default_value) where {L,U}
    if default_value === missing
        return missing
    end
    # Use round for converting floats to ints
    val = if isa(default_value, Integer)
        convert(Int, default_value)
    else
        round(Int, default_value)
    end
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    validate_bounds(val, lower, upper; kind="Int value")
    return val
end

function _create_with_default(
    ::Type{OfNamedTuple{Names,Types}}, default_value
) where {Names,Types}
    values = Tuple(
        _create_with_default(Types.parameters[i], default_value) for i in 1:length(Names)
    )
    return NamedTuple{Names}(values)
end

function _create_with_default(::Type{OfConstantWrapper{T}}, default_value) where {T}
    return error(
        "Cannot create values for constants. Provide the constant value in T(; const_name=value).",
    )
end

function _create_instance_impl(
    ::Type{T}, value_generator::Function, kwargs
) where {T<:OfNamedTuple}
    names = get_names(T)
    types = get_types(T)

    constants = Dict{Symbol,Any}()
    values = Dict{Symbol,Any}()

    for (key, val) in pairs(kwargs)
        idx = findfirst(==(key), names)
        if idx !== nothing && types.parameters[idx] <: OfConstantWrapper
            constants[key] = val
        else
            values[key] = val
        end
    end

    for (idx, name) in enumerate(names)
        if types.parameters[idx] <: OfConstantWrapper && !haskey(constants, name)
            error("Constant `$name` is required but not provided")
        end
    end

    # First concretize with constants
    concrete_type = of(T, NamedTuple(constants))

    if has_symbolic_dims(concrete_type)
        missing_symbols = get_unresolved_symbols(concrete_type)
        error("Missing values for symbolic dimensions: $(join(missing_symbols, ", "))")
    end

    # Get the names and types from the concrete type (constants removed)
    concrete_names = get_names(concrete_type)
    concrete_types = get_types(concrete_type)

    # Build the result with provided values or defaults
    result_values = Any[]
    for (idx, name) in enumerate(concrete_names)
        field_type = concrete_types.parameters[idx]

        if haskey(values, name)
            try
                push!(result_values, _validate(field_type, values[name]))
            catch e
                error("Validation failed for field $name: $(e.msg)")
            end
        else
            push!(result_values, value_generator(field_type))
        end
    end

    return NamedTuple{concrete_names}(Tuple(result_values))
end

"""
    (::Type{T})(; kwargs...) where {T<:OfNamedTuple}
    (::Type{T})(default_value; kwargs...) where {T<:OfNamedTuple}

Create an instance of an `OfNamedTuple` type with specified constant values.

This constructor creates actual values (instances) from an `OfNamedTuple` specification.
It requires all constants to be provided and initializes non-constant fields
to zero or a specified default value. Only `OfNamedTuple` types are constructible this
way; the leaf types (`OfReal`, `OfInt`, …) are specifications, not instantiable objects.

# Arguments
- `T<:OfNamedTuple`: The named-tuple specification to instantiate
- `default_value`: Optional value to initialize all non-constant fields (default: appropriate zero)
- `kwargs...`: Constant values and optionally non-constant field values

# Returns
A `NamedTuple` instance with all fields initialized.

# Examples
```julia
# Define type with constants
T = @of(
    n = of(Int; constant=true),
    mu = of(Real),
    data = of(Array, n, 2)
)

# Create instance with constants (non-constants default to zero)
instance = T(; n=5)
# Returns: (mu = 0.0, data = zeros(5, 2))

# Create instance with custom default
instance = T(1.0; n=5)  
# Returns: (mu = 1.0, data = ones(5, 2))

# Create instance with missing values
instance = T(missing; n=5)
# Returns: (mu = missing, data = 5×2 Array{Missing})

# Provide specific values for non-constants
instance = T(; n=5, mu=2.5, data=rand(5, 2))
# Returns: (mu = 2.5, data = <provided 5×2 array>)
```

# Errors
- Throws error if required constants are not provided

# See also
[`of`](@ref), [`@of`](@ref)
"""
function (::Type{T})(; kwargs...) where {T<:OfNamedTuple}
    return _create_instance_impl(T, zero, kwargs)
end

function (::Type{T})(default_value; kwargs...) where {T<:OfNamedTuple}
    return _create_instance_impl(
        T, field_type -> _create_with_default(field_type, default_value), kwargs
    )
end

"""
    @of(field1=spec1, field2=spec2, ...)

Create an `OfNamedTuple` type with cleaner syntax for field references.

The `@of` macro provides a more intuitive syntax for creating named tuple types
where fields can reference each other. Field names used in dimensions or bounds
are automatically converted to symbolic references.

# Syntax
```julia
@of(
    field_name = of_specification,
    ...
)
```

# Features
- Direct field references without quoting (e.g., `n` instead of `:n`)
- Support for arithmetic expressions in dimensions (e.g., `n+1`, `2*n`)
- Automatic conversion to appropriate `OfNamedTuple` type
- Fields are processed in order, allowing later fields to reference earlier ones

# Examples
```julia
# Basic usage with constants and arrays
T = @of(
    n = of(Int; constant=true),
    mu = of(Real),
    data = of(Array, n, 2)  # 'n' automatically converted to symbolic reference
)

# With arithmetic expressions
T = @of(
    n = of(Int; constant=true),
    original = of(Array, n, n),
    padded = of(Array, n+1, n+1),
    doubled = of(Array, 2*n, n)
)

# Nested structures: field references resolve within the @of that declares them, so keep
# a dimension's constants in the same block (cross-level paths like `dims.rows` are not
# supported), then concretize the whole tree at once.
Inner = @of(
    rows = of(Int; constant=true),
    cols = of(Int; constant=true),
    matrix = of(Array, rows, cols)
)
T = @of(block = Inner)
CT = of(T; rows=3, cols=4)
```

# See also
[`of`](@ref), [`OfNamedTuple`](@ref)
"""
macro of(args...)
    # Parse the arguments to extract field specifications
    fields = Dict{Symbol,Any}()
    field_order = Symbol[]

    for arg in args
        if !(arg isa Expr && arg.head == :(=) && length(arg.args) == 2)
            error("@of expects keyword arguments like field=spec")
        end

        field_name = arg.args[1]
        field_spec = arg.args[2]

        if field_name isa Symbol
            fields[field_name] = field_spec
            push!(field_order, field_name)
        else
            error("Field name must be a symbol, got $(field_name)")
        end
    end

    processed_fields = Dict{Symbol,Any}()

    for (field_name, spec) in fields
        processed_spec = process_of_spec(spec, field_order)
        processed_fields[field_name] = processed_spec
    end

    nt_expr = Expr(:tuple)
    for field_name in field_order
        push!(nt_expr.args, Expr(:(=), field_name, processed_fields[field_name]))
    end

    return esc(:(of($nt_expr)))
end

# Process an of specification, converting field references to symbols
function process_of_spec(spec::Expr, available_fields::Vector{Symbol})
    if spec.head == :call && length(spec.args) >= 1
        func = spec.args[1]

        # Check if this is an of(...) call
        if func == :of
            # Process the arguments
            new_args = Any[func]

            # Separate positional and keyword arguments
            pos_args = []
            kw_args = []

            for arg in spec.args[2:end]
                if arg isa Expr && arg.head == :parameters
                    # Handle parameters block (e.g., f(x; a=1, b=2))
                    for param in arg.args
                        push!(kw_args, param)
                    end
                elseif arg isa Expr && arg.head == :kw
                    # Handle individual keyword argument
                    push!(kw_args, arg)
                else
                    push!(pos_args, arg)
                end
            end

            # Process positional arguments
            for arg in pos_args
                processed_arg = process_dimension_arg(arg, available_fields)
                push!(new_args, processed_arg)
            end

            if !isempty(kw_args)
                params_expr = Expr(:parameters, kw_args...)
                insert!(new_args, 2, params_expr)
            end

            return Expr(:call, new_args...)
        else
            # Not an of call, process recursively
            return Expr(
                spec.head, [process_of_spec(arg, available_fields) for arg in spec.args]...
            )
        end
    else
        return spec
    end
end

process_of_spec(x, ::Vector{Symbol}) = x

# Process a dimension/bound argument, converting field references to symbols
function process_dimension_arg(arg, available_fields::Vector{Symbol})
    if arg isa Symbol && arg in available_fields
        # Convert field reference to symbol
        return QuoteNode(arg)
    elseif arg isa Expr
        return process_expression_refs(arg, available_fields)
    else
        return arg
    end
end

# Check if a processed expression is a `SymbolicExpr{...}` type. The head may be the bare
# symbol or the `GlobalRef` the macro emits for hygiene, so accept both.
function _is_symbolic_expr_type(expr::Expr)
    (expr.head == :curly && length(expr.args) >= 2) || return false
    head = expr.args[1]
    return head === :SymbolicExpr || (head isa GlobalRef && head.name === :SymbolicExpr)
end

# Process a single argument in an arithmetic expression
function _process_arithmetic_arg(arg, available_fields::Vector{Symbol})
    if arg isa Symbol && arg in available_fields
        return (QuoteNode(arg), true)
    elseif arg isa Expr
        processed = process_expression_refs(arg, available_fields)
        if _is_symbolic_expr_type(processed)
            # Extract the tuple from SymbolicExpr{...}
            return (processed.args[2], true)
        else
            return (processed, false)
        end
    else
        return (arg, false)
    end
end

# Process an arithmetic expression, converting field references to symbols
function _process_arithmetic_expr(expr::Expr, available_fields::Vector{Symbol})
    op = expr.args[1]

    # Build tuple representation: (op, arg1, arg2, ...)
    tuple_args = Any[QuoteNode(op)]
    has_field_ref = false

    for arg in expr.args[2:end]
        processed_arg, has_ref = _process_arithmetic_arg(arg, available_fields)
        push!(tuple_args, processed_arg)
        has_field_ref |= has_ref
    end

    if has_field_ref
        # Emit a fully-qualified `SymbolicExpr` so the escaped `@of` expansion resolves
        # even when the caller only does `using AbstractPPL` (the name is `public`, not exported).
        tuple_expr = Expr(:tuple, tuple_args...)
        return :($(GlobalRef(@__MODULE__, :SymbolicExpr)){$tuple_expr})
    else
        return expr
    end
end

# Process an expression, converting field references to symbols in expressions
function process_expression_refs(expr::Expr, available_fields::Vector{Symbol})
    # Check if this is an arithmetic call expression
    if expr.head == :call && length(expr.args) >= 2
        op = expr.args[1]
        if op in [:+, :-, :*, :/]
            return _process_arithmetic_expr(expr, available_fields)
        end
    end

    return expr
end

"""
    rand([rng::AbstractRNG], ::Type{T}) where T<:OfType

Generate random values matching the type specification.

Creates random instances that satisfy the constraints encoded in the `OfType`.
Arrays are filled with random values, named tuples recurse over their fields, and
bounded scalars respect their bounds. Pass an `rng` for reproducible draws; the
method without one uses `Random.default_rng()`.

The bounded/unbounded distributions are deliberate but unspecified conveniences:

- `OfReal`: uniform on `[lower, upper]`; for a single bound, a (reflected) shifted
  exponential; standard normal when unbounded.
- `OfInt`: uniform on `lower:upper`; for a single bound, an arbitrary 100-wide window
  anchored at it; `-100:100` when unbounded.
- `OfArray`: each element drawn as for its element type.
- `OfNamedTuple`: each field drawn recursively.
- `OfConstantWrapper`: not supported (throws).

# Examples
```julia
using Random

rand(of(Float64, 0, 1))                       # a Float64 in [0, 1]
rand(of(Array, 3, 4))                         # a 3×4 Matrix{Float64}
rand(StableRNG(1), @of(x=of(Real, 0, 1)))     # reproducible NamedTuple draw

T = @of(n=of(Int; constant=true), data=of(Array, n, 2))
rand(of(T; n=5))                              # (data = <random 5×2 array>,)
```

# Errors
- Throws for types with unresolved symbolic dimensions or bounds.
- Throws for constant wrapper types.

# See also
[`zero`](@ref), [`of`](@ref)
"""
Base.rand(::AbstractRNG, ::Type{<:OfType})

# Each type gets an explicit-RNG method (the contract downstream samplers rely on); the
# convenience method without an RNG forwards to `Random.default_rng()`.
function Base.rand(rng::AbstractRNG, ::Type{OfArray{T,N,D}}) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    any(_is_symbolic_dim, dims) && error(
        "Cannot generate random array with symbolic dimensions. Resolve them with of(T; name=value) first.",
    )
    return rand(rng, T, dims...)
end
Base.rand(::Type{OfArray{T,N,D}}) where {T,N,D} = rand(default_rng(), OfArray{T,N,D})

function Base.rand(rng::AbstractRNG, ::Type{OfReal{T,L,U}}) where {T,L,U}
    _assert_concrete_bounds(OfReal{T,L,U})
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    if !isnothing(lower) && !isnothing(upper)
        return T(lower + rand(rng) * (upper - lower))
    elseif !isnothing(lower)
        # Lower bound only: draw from a shifted exponential on [lower, ∞).
        return T(lower + randexp(rng))
    elseif !isnothing(upper)
        # Upper bound only: draw from a reflected shifted exponential on (-∞, upper].
        return T(upper - randexp(rng))
    else
        return T(randn(rng))
    end
end
Base.rand(::Type{OfReal{T,L,U}}) where {T,L,U} = rand(default_rng(), OfReal{T,L,U})

function Base.rand(rng::AbstractRNG, ::Type{OfInt{L,U}}) where {L,U}
    _assert_concrete_bounds(OfInt{L,U})
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    if !isnothing(lower) && !isnothing(upper)
        return rand(rng, lower:upper)
    elseif !isnothing(lower)
        # Lower bound only: an arbitrary but reasonable [lower, lower+100] window.
        return rand(rng, lower:(lower + 100))
    elseif !isnothing(upper)
        return rand(rng, (upper - 100):upper)
    else
        return rand(rng, -100:100)
    end
end
Base.rand(::Type{OfInt{L,U}}) where {L,U} = rand(default_rng(), OfInt{L,U})

# `@generated` so the per-field draws unroll and the NamedTuple eltypes stay inferable.
@generated function Base.rand(
    rng::AbstractRNG, ::Type{OfNamedTuple{Names,Types}}
) where {Names,Types}
    draws = [:(rand(rng, $P)) for P in Types.parameters]
    return :(NamedTuple{Names}(($(draws...),)))
end
function Base.rand(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    return rand(default_rng(), OfNamedTuple{Names,Types})
end

function Base.rand(::AbstractRNG, ::Type{OfConstantWrapper{T}}) where {T}
    return error(
        "Cannot generate random values for constants. Use rand(of(T; const_name=value)) after providing the constant value.",
    )
end
function Base.rand(::Type{OfConstantWrapper{T}}) where {T}
    return rand(default_rng(), OfConstantWrapper{T})
end

"""
    zero(::Type{T}) where T<:OfType

Generate zero/default values matching the type specification.

Creates instances initialized to appropriate zero values that satisfy the 
constraints. For bounded types where zero is outside the bounds, returns 
the nearest bound value.

# Behavior by Type
- `OfReal`: Returns 0.0 if within bounds, otherwise nearest bound
- `OfInt`: Returns 0 if within bounds, otherwise nearest bound  
- `OfArray`: Returns array filled with zeros
- `OfNamedTuple`: Recursively generates zero values for all fields
- `OfConstantWrapper`: Not supported (throws error)

# Examples
```julia
# Unbounded types
zero(of(Float64))           # 0.0
zero(of(Int))               # 0
zero(of(Array, 3, 2))       # 3×2 matrix of zeros

# Bounded types respect bounds
zero(of(Real, 1.0, 2.0))    # 1.0 (lower bound since 0 is outside)
zero(of(Int, -10, -5))      # -5 (upper bound since 0 is outside)

# Named tuples
T = @of(x=of(Real), y=of(Array, 2, 2))
zero(T)  # (x=0.0, y=[0.0 0.0; 0.0 0.0])

# With resolved constants
T = @of(n=of(Int; constant=true), data=of(Array, n, n))
zero(of(T; n=3))  # (data = 3×3 zero matrix)
```

# Errors
- Throws error for types with unresolved symbolic dimensions
- Throws error for constant wrapper types

# See also
[`rand`](@ref), [`of`](@ref)
"""
function Base.zero(::Type{OfArray{T,N,D}}) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    any(_is_symbolic_dim, dims) && error(
        "Cannot create zero array with symbolic dimensions. Resolve them with of(T; name=value) first.",
    )
    return zeros(T, dims...)
end

function Base.zero(::Type{OfReal{T,L,U}}) where {T,L,U}
    _assert_concrete_bounds(OfReal{T,L,U})
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    if !isnothing(lower) && lower > 0
        return T(lower)
    elseif !isnothing(upper) && upper < 0
        return T(upper)
    else
        return zero(T)
    end
end

function Base.zero(::Type{OfInt{L,U}}) where {L,U}
    _assert_concrete_bounds(OfInt{L,U})
    lower = type_to_bound(L)
    upper = type_to_bound(U)
    if !isnothing(lower) && lower > 0
        return lower
    elseif !isnothing(upper) && upper < 0
        return upper
    else
        return 0
    end
end

@generated function Base.zero(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    zeros_ = [:(zero($P)) for P in Types.parameters]
    return :(NamedTuple{Names}(($(zeros_...),)))
end

function Base.zero(::Type{OfConstantWrapper{T}}) where {T}
    return error(
        "Cannot generate zero values for constants. Use zero(of(T; const_name=value)) after providing the constant value.",
    )
end

"""
    size(::Type{T}) where T<:OfType

Get the dimensions/shape of an `OfType` specification.

Returns the size information encoded in the type. For arrays, returns a tuple
of dimensions. For scalars, returns an empty tuple. For named tuples, returns
a named tuple with the size of each field.

# Returns
- `OfArray`: Tuple of dimensions
- `OfReal`, `OfInt`: Empty tuple `()`
- `OfNamedTuple`: Named tuple with sizes of each field
- `OfConstantWrapper`: Delegates to wrapped type

# Examples
```julia
size(of(Array, 3, 4))        # (3, 4)
size(of(Float64))            # ()
size(of(Int, 0, 10))         # ()

T = @of(x=of(Real), y=of(Array, 2, 3))
size(T)                      # (x=(), y=(2, 3))
```

# Errors
- Throws error for arrays with unresolved symbolic dimensions

# See also
[`length`](@ref), [`of`](@ref)
"""
function Base.size(::Type{OfArray{T,N,D}}) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    any(_is_symbolic_dim, dims) &&
        error("Cannot get size of array with symbolic dimensions.")
    return dims
end

Base.size(::Type{OfReal{T,L,U}}) where {T,L,U} = ()
Base.size(::Type{OfInt{L,U}}) where {L,U} = ()

function Base.size(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    dims = ntuple(i -> size(Types.parameters[i]), length(Names))
    return NamedTuple{Names}(dims)
end

function Base.size(::Type{OfConstantWrapper{T}}) where {T}
    return size(T)
end

"""
    length(::Type{T}) where T<:OfType

Get the total number of elements when the type is flattened.

Returns the total count of numerical values that would be in a flattened
representation. Arrays contribute their total element count, scalars 
contribute 1, and named tuples sum the lengths of all fields. Constants
(wrapped in `OfConstantWrapper`) contribute 0 as they are not part of 
the flattened representation.

# Returns
- `OfArray`: Product of dimensions (total elements)
- `OfReal`, `OfInt`: 1
- `OfNamedTuple`: Sum of lengths of all fields
- `OfConstantWrapper`: 0 (constants excluded from flattening)

# Examples
```julia
length(of(Array, 3, 4))      # 12
length(of(Float64))          # 1
length(of(Int, 0, 10))       # 1

T = @of(x=of(Real), y=of(Array, 2, 3))
length(T)                    # 7 (1 + 6)

# Constants contribute 0; concrete fields still count
T2 = @of(n=of(Int; constant=true), data=of(Array, 3, 3))
length(T2)                   # 9 (n contributes 0, data contributes 9)
length(of(T2; n=5))          # 9 (n removed; data unchanged)
```

# Errors
- Throws error for arrays with unresolved symbolic dimensions

# See also
[`size`](@ref), [`flatten`](@ref), [`unflatten`](@ref)
"""
function Base.length(::Type{OfArray{T,N,D}}) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    any(_is_symbolic_dim, dims) &&
        error("Cannot get length of array with symbolic dimensions.")
    return prod(dims)::Int
end

Base.length(::Type{OfReal{T,L,U}}) where {T,L,U} = 1
Base.length(::Type{OfInt{L,U}}) where {L,U} = 1

# `@generated` so the total folds to a compile-time constant; a plain recursive `sum`
# widens to `Any` once nesting/field-count grows (poisoning the flatten/unflatten path).
@generated function Base.length(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    total = sum(Int[length(P) for P in Types.parameters]; init=0)
    return :($total)
end

function Base.length(::Type{OfConstantWrapper{T}}) where {T}
    return 0  # Constants are not part of the flattened representation
end

# Check if a type still carries unresolved symbols: symbolic array dimensions, symbolic
# bounds, or constants (an `OfConstantWrapper`, whether nested or at the top level).
function has_symbolic_dims(::Type{T}) where {T<:OfType}
    if T <: OfArray
        return any(_is_symbolic_dim, get_dims(T))
    elseif T <: Union{OfReal,OfInt}
        return _is_symbolic_bound(get_lower(T)) || _is_symbolic_bound(get_upper(T))
    elseif T <: OfConstantWrapper
        return true
    elseif T <: OfNamedTuple
        types = get_types(T)
        for i in 1:length(types.parameters)
            has_symbolic_dims(types.parameters[i]) && return true
        end
        return false
    else
        return false
    end
end

# Get list of unresolved symbols in a type
function get_unresolved_symbols(::Type{T}) where {T<:OfType}
    symbols = Symbol[]

    function collect_symbols(oft_type::Type, path::String="")
        if oft_type <: OfArray
            dims = get_dims(oft_type)
            for d in dims
                if d isa Symbol
                    push!(symbols, d)
                elseif d isa Type && d <: SymbolicExpr
                    extract_symbols_from_expr(d.parameters[1])
                end
            end
        elseif oft_type <: Union{OfReal,OfInt}
            for b in (get_lower(oft_type), get_upper(oft_type))
                if b isa Type && b <: SymbolicRef
                    push!(symbols, type_to_bound(b))
                elseif b isa Type && b <: SymbolicExpr
                    extract_symbols_from_expr(b.parameters[1])
                end
            end
        elseif oft_type <: OfConstantWrapper
            collect_symbols(get_wrapped_type(oft_type), path)
        elseif oft_type <: OfNamedTuple
            names = get_names(oft_type)
            types = get_types(oft_type)
            for (i, name) in enumerate(names)
                field_type = types.parameters[i]
                new_path = isempty(path) ? string(name) : "$path.$name"
                if field_type <: OfConstantWrapper
                    push!(symbols, name)
                else
                    collect_symbols(field_type, new_path)
                end
            end
        end
    end

    function extract_symbols_from_expr(expr::Tuple)
        for arg in expr[2:end]  # Skip operator
            if arg isa Symbol
                push!(symbols, arg)
            elseif arg isa Tuple
                extract_symbols_from_expr(arg)
            end
        end
    end

    collect_symbols(T)
    return unique(symbols)
end

# Validate that a value is within bounds. `kind` only labels the error message.
function validate_bounds(value, lower, upper; kind::AbstractString="value")
    if !isnothing(lower) && value < lower
        error("$kind $value is below lower bound $lower")
    end
    if !isnothing(upper) && value > upper
        error("$kind $value is above upper bound $upper")
    end
end

function _validate(::Type{T}, value) where {T<:OfType}
    if is_leaf(T)
        return _validate_leaf(T, value)
    else
        return _validate_container(T, value)
    end
end

function _validate_leaf(::Type{OfArray{T,N,D}}, value) where {T,N,D}
    value isa AbstractArray || error("Expected Array for OfArray, got $(typeof(value))")

    dims = get_dims(OfArray{T,N,D})
    any(d -> d isa Symbol || (d isa Type && d <: SymbolicExpr), dims) && error(
        "Cannot validate array with symbolic dimensions. Use the parameterized constructor.",
    )

    # Check dimensions before conversion
    ndims(value) == N ||
        error("Array dimension mismatch: expected $N dimensions, got $(ndims(value))")
    size(value) == Tuple(dims) ||
        error("Array size mismatch: expected $(Tuple(dims)), got $(size(value))")

    arr = convert(Array{T,N}, value)
    return arr
end

function _validate_leaf(::Type{OfReal{T,L,U}}, value) where {T,L,U}
    if value isa Real
        val = convert(T, value)
        lower = type_to_bound(L)
        upper = type_to_bound(U)
        validate_bounds(val, lower, upper; kind="Real value")
        return val
    else
        error("Expected Real for OfReal, got $(typeof(value))")
    end
end

function _validate_leaf(::Type{OfInt{L,U}}, value) where {L,U}
    if value isa Integer
        val = convert(Int, value)
        lower = type_to_bound(L)
        upper = type_to_bound(U)
        validate_bounds(val, lower, upper; kind="Int value")
        return val
    elseif value isa Real
        # Allow conversion from Real to Int if it's a whole number
        if isinteger(value)
            return _validate_leaf(OfInt{L,U}, Int(value))
        else
            error("Expected Integer for OfInt, got non-integer Real: $value")
        end
    else
        error("Expected Integer for OfInt, got $(typeof(value))")
    end
end

function _validate_container(::Type{OfNamedTuple{Names,Types}}, value) where {Names,Types}
    value isa NamedTuple ||
        error("Expected NamedTuple for OfNamedTuple, got $(typeof(value))")

    value_names = fieldnames(typeof(value))
    for name in Names
        if !(name in value_names)
            error("Missing required field: $name. Got fields: $(join(value_names, ", "))")
        end
    end

    vals = ntuple(length(Names)) do i
        field_name = Names[i]
        field_type = Types.parameters[i]
        _validate(field_type, getproperty(value, field_name))
    end
    return NamedTuple{Names}(vals)
end

function _validate_leaf(::Type{OfConstantWrapper{T}}, value) where {T}
    return _validate_leaf(T, value)
end

"""
    flatten(::Type{T}, values) where T<:OfType

Convert structured values to a flat numeric vector.

Walks `values` in field order, vectorising arrays (column-major) and recursing into
named tuples, and returns a flat vector whose element type is the promotion of the
declared leaf element types (so a pure-`Int` structure stays `Vector{Int}`, while any
float field widens the whole vector). This is the form an optimiser or sampler wants.
Constants are excluded by construction, since a flattenable type has none.

# Returns
A `Vector{V}` where `V` is `promote_type` of the declared leaf element types.

# Examples
```julia
flatten(of(Float64), 3.14)            # [3.14]            (Vector{Float64})
flatten(of(Array, 2, 2), [1 2; 3 4])  # [1.0, 3.0, 2.0, 4.0]

T = @of(x=of(Real), y=of(Array, 2, 2))
flatten(T, (x=1.5, y=[1 2; 3 4]))     # [1.5, 1.0, 3.0, 2.0, 4.0]
```

# Errors
- Throws if `values` do not match the specification (shape or bounds).
- Throws for types with unresolved symbolic dimensions, bounds, or constants.

# See also
[`unflatten`](@ref), [`length`](@ref)
"""
function flatten(::Type{T}, values) where {T<:OfType}
    has_symbolic_dims(T) && error(
        "Cannot flatten a type with symbolic dimensions, symbolic bounds, or constants. Resolve them with of(T; name=value) first.",
    )
    validated = _validate(T, values)
    out = Vector{_flat_eltype(T)}(undef, length(T))
    _fill_flat!(out, 1, T, validated)
    return out
end

# Element type of the flat vector: promotion of the declared leaf element types.
_flat_eltype(::Type{OfReal{T,L,U}}) where {T,L,U} = T
_flat_eltype(::Type{<:OfInt}) = Int
_flat_eltype(::Type{OfArray{T,N,D}}) where {T,N,D} = T
function _flat_eltype(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    return promote_type(map(_flat_eltype, (Types.parameters...,))...)
end

# Write a leaf/subtree into `out` starting at index `i`; return the next free index.
_fill_flat!(out, i, ::Type{<:OfReal}, x) = (out[i] = x; i + 1)
_fill_flat!(out, i, ::Type{<:OfInt}, x) = (out[i] = x; i + 1)
function _fill_flat!(out, i, ::Type{<:OfArray}, a)
    copyto!(out, i, vec(a), 1, length(a))
    return i + length(a)
end
# `@generated` so the per-field recursion unrolls and stays type-stable on the AD path.
@generated function _fill_flat!(
    out, i0, ::Type{OfNamedTuple{Names,Types}}, nt
) where {Names,Types}
    body = quote
        i = i0
    end
    for (k, P) in enumerate(Types.parameters)
        push!(body.args, :(i = _fill_flat!(out, i, $P, nt[$k])))
    end
    push!(body.args, :(return i))
    return body
end

"""
    unflatten(::Type{T}, flat_values::AbstractVector{<:Real}) where T<:OfType
    unflatten(::Type{T}, ::Missing) where T<:OfType

Reconstruct structured values from a flat numeric vector (the inverse of [`flatten`](@ref)).

Arrays are reshaped, named tuples are rebuilt in field order, and bounds are validated.
Floating-point leaves take `promote_type(declared, eltype(flat_values))`: the declared float
type acts as a precision floor, while wider numbers in `flat_values` — AD numbers
(`ForwardDiff.Dual`), `BigFloat` — flow through unchanged. Integer leaves are rounded to `Int`.

The `missing` method builds a structure with every element set to `missing`.

# Examples
```julia
unflatten(of(Float64), [3.14])            # 3.14
unflatten(of(Array, 2, 2), [1, 3, 2, 4])  # [1.0 2.0; 3.0 4.0]

T = @of(x=of(Real), y=of(Array, 2, 2))
unflatten(T, [1.5, 1.0, 3.0, 2.0, 4.0])   # (x=1.5, y=[1.0 2.0; 3.0 4.0])

unflatten(T, missing)                     # (x=missing, y=[missing missing; missing missing])
```

# Errors
- Throws if `length(flat_values)` differs from `length(T)`.
- Throws if values violate bounds.
- Throws for types with unresolved symbolic dimensions, bounds, or constants.

# See also
[`flatten`](@ref), [`length`](@ref)
"""
function unflatten(::Type{T}, flat_values::AbstractVector{<:Real}) where {T<:OfType}
    has_symbolic_dims(T) && error(
        "Cannot unflatten a type with symbolic dimensions, symbolic bounds, or constants. Resolve them with of(T; name=value) first.",
    )
    n = length(T)
    length(flat_values) == n ||
        error("Length mismatch: type expects $n values, got $(length(flat_values)).")
    value, _ = _unflat(T, flat_values, 1)
    return value
end

# Reconstruct one leaf/subtree from `v` starting at index `i`; return (value, next index).
# Float leaves take `promote_type(declared, eltype(v))`, so the declared type is a precision
# floor while AD numbers (`Dual`), `BigFloat`, etc. in `v` flow through.
function _unflat(::Type{OfReal{T,L,U}}, v, i) where {T,L,U}
    x = convert(promote_type(T, eltype(v)), v[i])
    validate_bounds(x, type_to_bound(L), type_to_bound(U); kind="Real value")
    return x, i + 1
end
function _unflat(::Type{OfInt{L,U}}, v, i) where {L,U}
    x = round(Int, v[i])
    validate_bounds(x, type_to_bound(L), type_to_bound(U); kind="Int value")
    return x, i + 1
end
function _unflat(::Type{OfArray{T,N,D}}, v, i) where {T,N,D}
    dims = get_dims(OfArray{T,N,D})
    n = prod(dims)
    arr = _to_array(T, @view(v[i:(i + n - 1)]), dims)
    return arr, i + n
end
@generated function _unflat(::Type{OfNamedTuple{Names,Types}}, v, i0) where {Names,Types}
    body = quote
        i = i0
    end
    syms = Symbol[]
    for (k, P) in enumerate(Types.parameters)
        s = Symbol(:val_, k)
        push!(syms, s)
        push!(body.args, :(($s, i) = _unflat($P, v, i)))
    end
    push!(body.args, :(return NamedTuple{Names}(($(syms...),)), i))
    return body
end

# Integer element types round to that type; otherwise promote the declared element type with
# the flat vector's eltype (honour the declaration, but let AD/wider numbers through). `collect`
# copies, so the result never aliases the input vector.
_to_array(::Type{ET}, slice, dims) where {ET<:Integer} = reshape(round.(ET, slice), dims)
function _to_array(::Type{ET}, slice, dims) where {ET}
    return reshape(collect(promote_type(ET, eltype(slice)), slice), dims)
end

function unflatten(::Type{T}, ::Missing) where {T<:OfType}
    has_symbolic_dims(T) && error(
        "Cannot unflatten a type with symbolic dimensions, symbolic bounds, or constants. Resolve them with of(T; name=value) first.",
    )
    return _unflat_missing(T)
end

_unflat_missing(::Type{<:OfReal}) = missing
_unflat_missing(::Type{<:OfInt}) = missing
function _unflat_missing(::Type{OfArray{T,N,D}}) where {T,N,D}
    return fill(missing, get_dims(OfArray{T,N,D})...)
end
@generated function _unflat_missing(::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    vals = [:(_unflat_missing($P)) for P in Types.parameters]
    return :(NamedTuple{Names}(($(vals...),)))
end

# Format a bound type for display
function format_bound(bound_type, constant_fields, use_color)
    if bound_type === Nothing
        return "nothing"
    elseif bound_type isa Real
        # Numeric values
        return string(bound_type)
    elseif bound_type isa Type && bound_type <: SymbolicRef
        sym = type_to_bound(bound_type)
        if sym in constant_fields && use_color
            return sprint() do io_inner
                printstyled(io_inner, string(sym); color=:cyan)
            end
        else
            return string(sym)
        end
    else
        return string(bound_type)
    end
end

# Show a bounded type (OfReal or OfInt) with proper formatting
function show_bounded_type(io::IO, type_name::String, L, U; constant::Bool=false)
    use_color = get(io, :color, false)
    constant_fields = get(io, :constant_fields, Symbol[])

    if L === Nothing && U === Nothing
        if constant
            if use_color
                printstyled(io, "of($type_name"; color=:cyan)
                printstyled(io, "; constant=true"; color=:light_black)
                printstyled(io, ")"; color=:cyan)
            else
                print(io, "of($type_name; constant=true)")
            end
        else
            print(io, "of($type_name)")
        end
    else
        lower_str = format_bound(L, constant_fields, use_color)
        upper_str = format_bound(U, constant_fields, use_color)

        if constant && use_color
            printstyled(io, "of($type_name, "; color=:cyan)
            if L isa Type && L <: SymbolicRef && type_to_bound(L) in constant_fields
                printstyled(io, string(type_to_bound(L)); color=:cyan)
            else
                printstyled(io, lower_str; color=:cyan)
            end
            printstyled(io, ", "; color=:cyan)
            if U isa Type && U <: SymbolicRef && type_to_bound(U) in constant_fields
                printstyled(io, string(type_to_bound(U)); color=:cyan)
            else
                printstyled(io, upper_str; color=:cyan)
            end
            printstyled(io, "; constant=true"; color=:light_black)
            printstyled(io, ")"; color=:cyan)
        else
            print(io, "of($type_name, ")
            if L isa Type &&
                L <: SymbolicRef &&
                type_to_bound(L) in constant_fields &&
                use_color
                printstyled(io, string(type_to_bound(L)); color=:cyan)
            else
                print(io, lower_str)
            end
            print(io, ", ")
            if U isa Type &&
                U <: SymbolicRef &&
                type_to_bound(U) in constant_fields &&
                use_color
                printstyled(io, string(type_to_bound(U)); color=:cyan)
            else
                print(io, upper_str)
            end
            if constant
                print(io, "; constant=true")
            end
            print(io, ")")
        end
    end
end

# Helper to convert expression tuple back to string
function expr_tuple_to_string(expr::Tuple)
    if length(expr) < 2
        return string(expr)
    end

    op = expr[1]
    if op in (:+, :-, :*, :/) && length(expr) == 3
        arg1_str = format_expr_arg(expr[2])
        arg2_str = format_expr_arg(expr[3])

        # Add parentheses for multiplication and division if needed
        if op in (:*, :/) && expr[2] isa Tuple
            arg1_str = "($arg1_str)"
        end
        if op in (:*, :/) && expr[3] isa Tuple
            arg2_str = "($arg2_str)"
        end

        return "$arg1_str $op $arg2_str"
    else
        return string(expr)
    end
end

# Format a single expression argument
function format_expr_arg(arg)
    if arg isa Symbol
        string(arg)
    elseif arg isa Tuple
        expr_tuple_to_string(arg)
    else
        string(arg)
    end
end

# Show implementations. Each guards against non-concrete types (free TypeVars, e.g. when a
# method signature or stacktrace frame is rendered): touching the static params there throws
# UndefVarError, which is fatal mid-backtrace, so fall back to Base's generic Type printer.
function Base.show(io::IO, t::Type{OfArray{T,N,D}}) where {T,N,D}
    isconcretetype(t) || return invoke(show, Tuple{IO,Type}, io, t)
    use_color = get(io, :color, false)
    constant_fields = get(io, :constant_fields, Symbol[])

    # Process dimensions, highlighting those that reference constants
    if use_color && !isempty(constant_fields)
        print(io, "of(Array, ")
        if T !== Float64
            print(io, T, ", ")
        end
        # D is a Tuple type, so we need to access its parameters
        dims_list = get_dims(OfArray{T,N,D})
        for (i, d) in enumerate(dims_list)
            if d isa Symbol && d in constant_fields
                printstyled(io, string(d); color=:cyan)
            elseif d isa Type && d <: SymbolicExpr
                # This is an expression - format it nicely
                expr = d.parameters[1]
                expr_str = expr_tuple_to_string(expr)
                # Check if any symbols in the expression are constants
                has_constant = false
                function check_expr(e::Tuple)
                    for arg in e[2:end]
                        if arg isa Symbol && arg in constant_fields
                            has_constant = true
                        elseif arg isa Tuple
                            check_expr(arg)
                        end
                    end
                end
                check_expr(expr)
                if has_constant
                    printstyled(io, expr_str; color=:cyan)
                else
                    print(io, expr_str)
                end
            else
                print(io, string(d))
            end
            if i < length(dims_list)
                print(io, ", ")
            end
        end
        print(io, ")")
        return nothing
    end

    # Non-color version.
    dims_list = get_dims(OfArray{T,N,D})
    dims_str = join(
        map(dims_list) do d
            if d isa Type && d <: SymbolicExpr
                expr_tuple_to_string(d.parameters[1])
            else
                string(d)
            end
        end,
        ", ",
    )

    prefix = T === Float64 ? "of(Array" : "of(Array, $T"
    # Append dims only when present, so a 0-dim array prints `of(Array)` not `of(Array, )`.
    return print(io, isempty(dims_str) ? "$prefix)" : "$prefix, $dims_str)")
end

function Base.show(io::IO, t::Type{OfReal{T,L,U}}) where {T,L,U}
    isconcretetype(t) || return invoke(show, Tuple{IO,Type}, io, t)
    return show_bounded_type(io, string(T), L, U)
end

function Base.show(io::IO, t::Type{OfInt{L,U}}) where {L,U}
    isconcretetype(t) || return invoke(show, Tuple{IO,Type}, io, t)
    return show_bounded_type(io, "Int", L, U)
end

# Helper function to collect constant fields from a NamedTuple type
function _collect_constant_fields(Names, Types)
    constant_fields = Symbol[]
    for (name, T) in zip(Names, Types.parameters)
        if T <: OfConstantWrapper
            push!(constant_fields, name)
        end
    end
    return constant_fields
end

# Helper function to estimate output length for a NamedTuple type
function _estimate_namedtuple_length(Names, Types)
    total_length = 4  # "@of("
    for (name, T) in zip(Names, Types.parameters)
        total_length += length(string(name)) + 1  # name=
        # Rough estimate of type string length
        if T <: OfArray
            dims = get_dims(T)
            total_length += 15 + sum(d -> length(string(d)), dims; init=0)
        else
            total_length += 20
        end
        total_length += 2  # ", "
    end
    return total_length
end

# Helper function to show a field with appropriate styling
function _show_field_type(io::IO, T::Type, is_constant::Bool)
    if is_constant && get(io, :color, false)
        wrapped = get_wrapped_type(T)
        if wrapped <: OfReal &&
            get_lower(wrapped) === Nothing &&
            get_upper(wrapped) === Nothing
            elem_type = get_element_type(wrapped)
            type_name = elem_type === Float64 ? "Real" : string(elem_type)
            printstyled(io, "of($type_name"; color=:cyan)
            printstyled(io, "; constant=true"; color=:light_black)
            printstyled(io, ")"; color=:cyan)
        elseif wrapped <: OfInt &&
            get_lower(wrapped) === Nothing &&
            get_upper(wrapped) === Nothing
            printstyled(io, "of(Int"; color=:cyan)
            printstyled(io, "; constant=true"; color=:light_black)
            printstyled(io, ")"; color=:cyan)
        else
            show(io, T)
        end
    else
        show(io, T)
    end
end

# Helper function to print a single field in NamedTuple
function _print_namedtuple_field(
    io::IO, name::Symbol, T::Type, is_constant::Bool, separator::String
)
    if is_constant && get(io, :color, false)
        printstyled(io, name; color=:cyan, bold=true)
    else
        print(io, name)
    end

    print(io, separator)
    return _show_field_type(io, T, is_constant)
end

function Base.show(io::IO, t::Type{OfNamedTuple{Names,Types}}) where {Names,Types}
    isconcretetype(t) || return invoke(show, Tuple{IO,Type}, io, t)
    # Collect constant fields to pass to child types
    constant_fields = _collect_constant_fields(Names, Types)
    io_with_constants = IOContext(io, :constant_fields => constant_fields)

    compact = get(io, :compact, false)
    multiline = !compact && length(Names) > 3

    # Check if single-line output would be too long
    if !multiline && !compact
        multiline = _estimate_namedtuple_length(Names, Types) > 80
    end

    print(io, "@of(")

    if multiline
        println(io)
        for (i, (name, T)) in enumerate(zip(Names, Types.parameters))
            print(io, "    ")
            is_constant = T <: OfConstantWrapper
            _print_namedtuple_field(io_with_constants, name, T, is_constant, " = ")

            if i < length(Names)
                println(io, ",")
            else
                println(io)
            end
        end
        print(io, ")")
    else
        # Single line format
        for (i, (name, T)) in enumerate(zip(Names, Types.parameters))
            is_constant = T <: OfConstantWrapper
            _print_namedtuple_field(io_with_constants, name, T, is_constant, "=")

            if i < length(Names)
                print(io, ", ")
            end
        end
        print(io, ")")
    end
end

function Base.show(io::IO, t::Type{OfConstantWrapper{T}}) where {T}
    isconcretetype(t) || return invoke(show, Tuple{IO,Type}, io, t)
    # Show the wrapped type with constant=true
    if T <: OfReal
        elem_type = get_element_type(T)
        # Use "Real" for backward compatibility when Float64 is the element type
        type_name = elem_type === Float64 ? "Real" : string(elem_type)
        show_bounded_type(io, type_name, get_lower(T), get_upper(T); constant=true)
    elseif T <: OfInt
        show_bounded_type(io, "Int", get_lower(T), get_upper(T); constant=true)
    elseif T <: OfArray
        # This case should not happen since constant=true is not allowed for Arrays
        # But if it does, show it as a fallback
        print(io, "OfConstantWrapper{", T, "}")
        printstyled(io, " # Invalid: constant=true not supported for Arrays"; color=:red)
    else
        # Fallback
        print(io, "OfConstantWrapper{", T, "}")
    end
end
