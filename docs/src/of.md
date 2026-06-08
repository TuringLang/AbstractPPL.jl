# The `of` Type System

## Overview

The `of` type system provides a declarative way to specify parameter **types** for
probabilistic programming. It is a lightweight, framework-agnostic type-annotation
system that:

  - Returns actual Julia types (not instances) that can be used in type annotations
  - Encodes specifications (dimensions, bounds) in type parameters
  - Provides utilities for parameter manipulation (`rand`, `zero`, `flatten`, `unflatten`)

It lives in AbstractPPL so that downstream packages can share a common vocabulary for
describing the shape, element type, and support of model variables. JuliaBUGS, for
example, uses it for `@model` parameter annotations.

## Core Concepts

### 1. Type-Based Design

The `of` function returns types with specifications encoded in type parameters:

  - `of(Array, dims...)` → `OfArray{Float64, N, (dim1, dim2, ...)}` - Arrays with specified dimensions
  - `of(Array, T, dims...)` → `OfArray{T, N, (dim1, dim2, ...)}` - Typed arrays
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

### 2. Type Parameter Encoding

The system encodes extra useful information into type parameters:

  - **Dimensions**: Stored as tuple type parameters (e.g., `(3, 4)` for a 3×4 matrix)
  - **Bounds**: Numeric literals stored directly as type parameters (e.g., `0.0`, `1.0`), or `Nothing` for unbounded
  - **Symbolic references**: Encoded using `SymbolicRef{:symbol}` for referencing other fields
  - **Arithmetic expressions**: Encoded using `SymbolicExpr{expr}` for expressions like `n+1`, `2*n`, etc. Division operations must result in integers for array dimensions.
  - **Field names**: Stored as a tuple of symbols in `OfNamedTuple`
  - **Element types**: Preserved as type parameters for arrays and nested structures

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

The `@of` macro provides cleaner syntax by automatically converting field references to symbols:

```julia
T = @of(
    n = of(Int; constant=true),
    data = of(Array, n, 2)  # 'n' is automatically converted to :n
)
```

### 5. Symbolic Dimensions and Bounds

For cases where dimensions need to be specified at runtime:

```julia
# Define type with symbolic dimensions using @of macro
MatrixType = @of(
    rows = of(Int; constant=true),
    cols = of(Int; constant=true),
    data = of(Array, rows, cols),
)

# Create concrete type by resolving constants
ConcreteType = of(MatrixType; rows=3, cols=4)
# ConcreteType is @of(data=of(Array, 3, 4))

# Use concrete type with rand and zero
rand(ConcreteType)  # generates random 3×4 matrix wrapped in NamedTuple
zero(ConcreteType)  # generates zero 3×4 matrix wrapped in NamedTuple

# Partial concretization (semiconcretized)
SemiConcreteType = of(MatrixType; rows=3)
# SemiConcreteType is @of(cols=of(Int; constant=true), data=of(Array, 3, :cols))

# Create instance by providing all constants (default to zero for data)
instance = MatrixType(; rows=3, cols=4)
# instance = (data = zeros(3, 4),)

# Create instance with missing values
instance = MatrixType(missing; rows=3, cols=4)
# instance = (data = (3×4 matrix of `missing`s),)

# Create instance with specific data
instance = MatrixType(; rows=3, cols=4, data=rand(3, 4))
# instance = (data = <provided 3×4 matrix>,)

# Create concrete type for flatten/unflatten (flatten/unflatten are public, not exported)
flat = AbstractPPL.flatten(ConcreteType, instance)
reconstructed = AbstractPPL.unflatten(ConcreteType, flat)

# rand and zero with concrete types
rand(of(MatrixType; rows=3, cols=4))  # generates random instance
zero(of(MatrixType; rows=10, cols=5)) # generates zero instance

# Missing constants will error
MatrixType(; rows=3) # Error: Constant `cols` is required but not provided
rand(MatrixType)     # Error: Cannot generate random values for types with symbolic dimensions
```

Arithmetic expressions in dimensions are also supported:

```julia
ExpandedMatrixType = @of(
    n = of(Int; constant=true),
    original = of(Array, n, n),
    padded = of(Array, n + 1, n + 1),
    doubled = of(Array, 2 * n, n),
    halved = of(Array, n / 2, n),
)

# Create instance - all non-constant fields default to zero
instance = ExpandedMatrixType(; n=10)
# This creates an instance with:
# - original: 10×10 zero matrix
# - padded: 11×11 zero matrix
# - doubled: 20×10 zero matrix
# - halved: 5×10 zero matrix  (n/2 must result in an integer, error if not)

# Create instance with custom default value
instance = ExpandedMatrixType(1.0; n=10)
# This creates an instance with all matrices filled with 1.0
```

## Flattening parameters

`flatten`/`unflatten` are useful for code that needs a flat parameter vector (for
example, an optimiser or a sampler) while keeping a structured view of the parameters:

```julia
using AbstractPPL: flatten, unflatten

Params = @of(mu = of(Real), sigma = of(Real, 0, nothing), beta = of(Array, Float64, 3),)

values = (mu=0.5, sigma=1.2, beta=[0.1, 0.2, 0.3])

flat = flatten(Params, values)          # length(Params) == 5
reconstructed = unflatten(Params, flat) # back to the (mu, sigma, beta) NamedTuple
```

`flatten` returns a vector whose element type is the promotion of the declared leaf types,
and `unflatten` is automatic-differentiation transparent: floating-point leaves take
`promote_type(declared, eltype(flat))`, so `ForwardDiff.Dual` (or `BigFloat`, …) numbers in
the flat vector flow through to the reconstructed structure. This makes the pair suitable for
gradient-based samplers and optimisers that differentiate through `unflatten`.

Constants (fields wrapped with `constant=true`) are excluded from the flattened
representation and must be resolved with `of(T; kwargs...)` before flattening.

## Use in models

Because `of` returns ordinary Julia types, the result can be used directly as a type
annotation. Downstream packages build on this: JuliaBUGS, for instance, accepts an `of`
type as the parameter annotation of a `@model`'s argument destructuring, e.g.
`(; mu, beta, sigma)::ParamsType`. See the JuliaBUGS documentation for the modelling
integration.

## API Reference

```@docs
of
@of
AbstractPPL.OfType
AbstractPPL.OfReal
AbstractPPL.OfInt
AbstractPPL.OfArray
AbstractPPL.OfNamedTuple
AbstractPPL.OfConstantWrapper
AbstractPPL.SymbolicRef
AbstractPPL.SymbolicExpr
```
