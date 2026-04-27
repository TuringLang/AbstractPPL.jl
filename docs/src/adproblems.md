# Evaluator preparation and AD

AbstractPPL provides a small interface for preparing callables and asking a
prepared evaluator for values and derivatives. `prepare` binds a callable to a
sample input that establishes the expected input shape and type;
`value_and_gradient` and `value_and_jacobian` then return the value and
derivative together.

## Quick start

```@example ad
using AbstractPPL
using AbstractPPL: prepare, value_and_gradient, value_and_jacobian

# 1. Define your problem as any callable.
struct MyModel end
(::MyModel)(x::AbstractVector) = sum(abs2, x)

# 2. Define the prepared evaluator shape for that problem.
struct MyPrepared{F}
    f::F
end

function AbstractPPL.prepare(model::MyModel, x::AbstractVector{<:Real})
    return MyPrepared(model)
end

(prepared::MyPrepared)(x::AbstractVector) = prepared.f(x)

function AbstractPPL.value_and_gradient(
    prepared::MyPrepared, x::AbstractVector{<:Real}
)
    return (prepared(x), 2 .* x)
end

# 3. Prepare and evaluate.
prepared = prepare(MyModel(), zeros(3))
value_and_gradient(prepared, [1.0, 2.0, 3.0])
```

## Two input styles

### Vector inputs

When the callable accepts a flat vector, pass a sample vector whose length
matches the expected input:

```@example ad
prepared_vec = prepare(MyModel(), zeros(3))
prepared_vec([1.0, 2.0, 3.0])
```

For vector-valued callables, use `value_and_jacobian`. The returned Jacobian
has shape `(length(value), length(x))`:

```@example ad
struct VecModel end
(::VecModel)(x::AbstractVector) = [x[1] * x[2], x[2] + x[3]]

struct VecPrepared{F}
    f::F
end

function AbstractPPL.prepare(model::VecModel, x::AbstractVector{<:Real})
    return VecPrepared(model)
end

(prepared::VecPrepared)(x::AbstractVector) = prepared.f(x)

function AbstractPPL.value_and_jacobian(
    prepared::VecPrepared, x::AbstractVector{<:Real}
)
    return (prepared(x), [x[2] x[1] 0; 0 1 1])
end

prepared_jac = prepare(VecModel(), zeros(3))
value_and_jacobian(prepared_jac, [2.0, 3.0, 4.0])
```

### NamedTuple inputs

When the callable accepts a `NamedTuple`, pass a sample `NamedTuple` whose
field names and value types match the expected input:

```@example ad
struct NTModel end
function AbstractPPL.prepare(::NTModel, values::NamedTuple)
    return (values::NamedTuple) -> values.a^2 + sum(abs2, values.b)
end

nt_model = NTModel()
nt0 = (a=0.0, b=zeros(2))
prepared_nt = prepare(nt_model, nt0)
prepared_nt((a=1.0, b=[2.0, 3.0]))
```

## AD backends

Automatic differentiation packages can extend the interface with
backend-specific three-argument methods:

```julia
prepared = prepare(adtype, problem, prototype)
value_and_gradient(prepared, x)
value_and_jacobian(prepared, x)
```

This branch provides a `DifferentiationInterface` extension. Load
`DifferentiationInterface` together with a backend supported by that package to
activate vector-input AD preparation for `ADTypes.AbstractADType` backends.

## Without an AD backend

The two-argument form `prepare(problem, x)` is available without any AD package.
It returns the callable unchanged by default, so code that calls `prepare`
unconditionally works regardless of which backends are loaded:

```@example ad
struct SimpleProblem end
(::SimpleProblem)(x::AbstractVector) = sum(x)

p = prepare(SimpleProblem(), zeros(3))
p([1.0, 2.0, 3.0])
```

## Supported extensions

| Extension                                | Trigger package            | Notes                                                               |
|:---------------------------------------- |:-------------------------- |:------------------------------------------------------------------- |
| `AbstractPPLDifferentiationInterfaceExt` | `DifferentiationInterface` | Vector inputs for compatible `ADTypes.AbstractADType` backend types |

## API reference

```@docs
AbstractPPL.prepare
AbstractPPL.value_and_gradient
AbstractPPL.value_and_jacobian
```
