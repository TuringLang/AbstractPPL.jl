# The `of` Type System

## Overview

The `of` type system provides a declarative way to specify parameter **types** for
probabilistic programming. It is a lightweight, framework-agnostic type-annotation
system that:

  - Returns schema types (not instances) for downstream annotation systems
  - Encodes specifications (dimensions, bounds) in type parameters
  - Provides utilities for parameter manipulation (`rand`, `zero`, `flatten`, `unflatten`)

It lives in AbstractPPL so that downstream packages can share a common vocabulary for
describing the shape, element type, and support of model variables. JuliaBUGS, for
example, uses it for `@model` parameter annotations.

The examples on this page are executed when the documentation is built. The imports are
brought into scope here; later examples reuse them.

```@setup of
using AbstractPPL
using AbstractPPL: flatten, unflatten
using Random
```

```@example of
using AbstractPPL
using AbstractPPL: flatten, unflatten
using Random
nothing # hide
```

## Core Concepts

### 1. Type-Based Design

The `of` function returns types with specifications encoded in type parameters:

  - `of(Array, dims...)` → `OfArray{Float64, N, (dim1, dim2, ...)}` - Arrays with specified dimensions
  - `of(Array, T, dims...)` → `OfArray{T, N, (dim1, dim2, ...)}` - Typed numeric arrays (`T <: Number`)
  - `of(Float64)` → `OfReal{Float64, Nothing, Nothing}` - Unbounded 64-bit floating point numbers
  - `of(Float32)` → `OfReal{Float32, Nothing, Nothing}` - Unbounded 32-bit floating point numbers
  - `of(Float64, lower, upper)` → `OfReal{Float64, lower, upper}` - Bounded 64-bit floats
  - `of(Float32, lower, upper)` → `OfReal{Float32, lower, upper}` - Bounded 32-bit floats
  - `of(Real)` → `OfReal{Float64, Nothing, Nothing}` - Unbounded real numbers (defaults to Float64)
  - `of(Real, lower, upper)` → `OfReal{Float64, lower, upper}` - Bounded real numbers (defaults to Float64)
  - `of(Int)` → `OfInt{Nothing, Nothing}` - Unbounded integers
  - `of(Int, lower, upper)` → `OfInt{lower, upper}` - Bounded integers
  - `@of(field1=..., field2=...)` → `OfNamedTuple{(:field1, :field2), Tuple{Type1, Type2}}` - Named tuples (use `@of` macro)
  - `of(...; constant=true)` → `OfConstantWrapper{T}` - Marks a type as constant/hyperparameter (supported for float types and `Int`)

A few `of(...)` calls and the concrete types they return:

```@example of
of(Float64, 0, 1)
```

```@example of
of(Array, 3, 4)
```

```@example of
of(Int; constant=true)
```

### 2. Type Parameter Encoding

The system encodes extra useful information into type parameters:

  - **Dimensions**: Stored as tuple type parameters (e.g., `(3, 4)` for a 3×4 matrix)
  - **Bounds**: Numeric literals stored directly as type parameters (e.g., `0.0`, `1.0`), or `Nothing` for unbounded
  - **Symbolic references**: Encoded using `SymbolicRef{:symbol}` for referencing earlier constant fields
  - **Arithmetic expressions**: Encoded using `SymbolicExpr{expr}` for expressions like `n+1`, `2*n`, etc. Division operations must result in integers for array dimensions.
  - **Field names**: Stored as a tuple of symbols in `OfNamedTuple`
  - **Element types**: Preserved as type parameters for numeric arrays and nested structures

### 3. Operations on Types

  - `T(; kwargs...)` where `T<:OfNamedTuple` — Create instances with specified constants (returns values, not types). Uses `zero()` as the default for missing values.

  - `T(default_value; kwargs...)` where `T<:OfNamedTuple` — Create instances with specified constants and initialise all element values to `default_value`, e.g. `T(missing; kwargs...)` initialises all element values to `missing`. `T(...)` returns instances, not types.
  - `of(T; kwargs...)` where `T<:OfType` — Create concrete types by resolving constants
  - `rand([rng], T::Type{<:OfType})` — Generate random values matching the type specification (pass an `AbstractRNG` for reproducible draws)
  - `zero(T::Type{<:OfType})` — Generate zero/default values
  - `size(T::Type{<:OfType})` — Get the dimensions/shape of the type
  - `length(T::Type{<:OfType})` — Get the total number of elements when flattened
  - `flatten(T::Type{<:OfType}, values)` — Convert structured values to a flat vector (element type is the promotion of the declared leaf types)
  - `unflatten(T::Type{<:OfType}, vec)` — Reconstruct structured values from a flat vector (float leaves take `promote_type(declared, eltype(vec))`, so AD numbers flow through)
  - `unflatten(T::Type{<:OfType}, missing)` — Create instances where element values are initialised to `missing`

Only `of` and `@of` are exported. `flatten`, `unflatten`, the `OfType` subtypes, and the
inspection helpers are `public` but not exported, so qualify them (`AbstractPPL.flatten`) or
bring them into scope with `using AbstractPPL: flatten, unflatten`.

### 4. The `@of` Macro

The `@of` macro provides cleaner syntax by automatically converting references to earlier
constant fields to symbols. Here `n` in the array dimension is automatically converted to
the symbol `:n`:

```@example of
T = @of(
    n = of(Int; constant=true),
    data = of(Array, n, 2)  # 'n' is automatically converted to :n
)
```

### 5. Symbolic Dimensions and Bounds

For cases where dimensions need to be specified at runtime, declare the dimensions as
constants and reference them in the array specifications:

```@example of
MatrixType = @of(
    rows = of(Int; constant=true),
    cols = of(Int; constant=true),
    data = of(Array, rows, cols),
)
```

Resolving the constants with `of(MatrixType; ...)` produces a concrete type with the
symbolic dimensions filled in:

```@example of
ConcreteType = of(MatrixType; rows=3, cols=4)
```

The concrete type works with [`rand`](@ref) and [`zero`](@ref). The draw uses a seeded RNG
so the rendered output is reproducible:

```@example of
rand(MersenneTwister(0), ConcreteType)  # random 3×4 matrix wrapped in a NamedTuple
```

```@example of
zero(ConcreteType)  # zero 3×4 matrix wrapped in a NamedTuple
```

Concretization can be partial. Resolving only `rows` leaves `cols` symbolic
(semiconcretized):

```@example of
SemiConcreteType = of(MatrixType; rows=3)
```

Calling the type as a constructor builds an instance. With all constants provided, the
non-constant `data` field defaults to zeros:

```@example of
MatrixType(; rows=3, cols=4)
```

Passing `missing` initialises element values to `missing`:

```@example of
MatrixType(missing; rows=3, cols=4)
```

Specific data can be provided directly for non-constant fields:

```@example of
MatrixType(; rows=3, cols=4, data=ones(3, 4))
```

A concrete type can be flattened and reconstructed. Here we flatten a `3×4` instance and
recover it (`flatten`/`unflatten` are public, not exported):

```@example of
instance = MatrixType(; rows=3, cols=4)
flat = flatten(ConcreteType, instance)
```

```@example of
reconstructed = unflatten(ConcreteType, flat)
```

`rand` and `zero` also work directly on a concretized type:

```@example of
rand(MersenneTwister(0), of(MatrixType; rows=3, cols=4))  # random instance
```

```@example of
zero(of(MatrixType; rows=10, cols=5))  # zero instance
```

Operations that still need unresolved information error. Constructing with a missing
constant throws, so we catch and display the message:

```@example of
try
    MatrixType(; rows=3)  # `cols` is required but not provided
catch err
    showerror(stdout, err)
end
```

Likewise, drawing from a type with unresolved symbolic dimensions throws:

```@example of
try
    rand(MatrixType)  # symbolic dimensions are unresolved
catch err
    showerror(stdout, err)
end
```

#### Arithmetic expressions in dimensions

Dimensions may be arithmetic expressions of constant fields. Division operations must
result in integers for array dimensions:

```@example of
ExpandedMatrixType = @of(
    n = of(Int; constant=true),
    original = of(Array, n, n),
    padded = of(Array, n + 1, n + 1),
    doubled = of(Array, 2 * n, n),
    halved = of(Array, n / 2, n),
)
```

Creating an instance with `n=10` evaluates each expression: `original` is `10×10`,
`padded` is `11×11`, `doubled` is `20×10`, and `halved` is `5×10`. Non-constant fields
default to zero. We display each field's shape:

```@example of
instance = ExpandedMatrixType(; n=10)
map(size, instance)
```

A custom default value fills every matrix instead of using zeros:

```@example of
instance = ExpandedMatrixType(1.0; n=10)
instance.original
```

If a division does not yield an integer dimension, instantiation throws. With `n=9`,
`n / 2 = 4.5` is not an integer:

```@example of
try
    ExpandedMatrixType(; n=9)  # n / 2 = 4.5 is not an integer
catch err
    showerror(stdout, err)
end
```

## Flattening parameters

`flatten`/`unflatten` are useful for code that needs a flat parameter vector (for
example, an optimiser or a sampler) while keeping a structured view of the parameters.
We define a small parameter specification:

```@example of
Params = @of(mu = of(Real), sigma = of(Real, 0, nothing), beta = of(Array, Float64, 3))
```

The total flattened length is `length(Params)`:

```@example of
length(Params)
```

Flattening a structured value produces a flat vector:

```@example of
values = (mu=0.5, sigma=1.2, beta=[0.1, 0.2, 0.3])
flat = flatten(Params, values)
```

`unflatten` reconstructs the original `(mu, sigma, beta)` NamedTuple:

```@example of
reconstructed = unflatten(Params, flat)
```

`flatten` returns a vector whose element type is the promotion of the declared leaf types,
and `unflatten` is automatic-differentiation transparent: floating-point leaves take
`promote_type(declared, eltype(flat))`, so `ForwardDiff.Dual` (or `BigFloat`, …) numbers in
the flat vector flow through to the reconstructed structure. This makes the pair suitable for
gradient-based samplers and optimisers that differentiate through `unflatten`.

Constants (fields wrapped with `constant=true`) are excluded from the flattened
representation and must be resolved with `of(T; kwargs...)` before flattening.

## Use in models

Because `of` returns schema types, downstream packages can use those types in their own
annotation systems. JuliaBUGS, for instance, accepts an `of` type as the parameter
annotation of a `@model`'s argument destructuring, e.g. `(; mu, beta, sigma)::ParamsType`.
These schema types are not supertypes of raw values, so `1.0 isa of(Float64)` is false;
see the downstream package documentation for the modelling integration.

## API Reference

```@docs
of
@of
AbstractPPL.flatten
AbstractPPL.unflatten
Base.rand(::Random.AbstractRNG, ::Type{<:AbstractPPL.OfType})
Base.zero
Base.size
Base.length
AbstractPPL.OfType
AbstractPPL.OfReal
AbstractPPL.OfInt
AbstractPPL.OfArray
AbstractPPL.OfNamedTuple
AbstractPPL.OfConstantWrapper
AbstractPPL.SymbolicRef
AbstractPPL.SymbolicExpr
```
