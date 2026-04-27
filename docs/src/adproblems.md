# Evaluator preparation and AD

AbstractPPL provides a small interface for preparing any callable to be
differentiated by an AD backend. `prepare` binds a callable to a sample input
that establishes the expected input shape and type; `value_and_gradient` and
`value_and_jacobian` then return the value and derivative together.

## Quick start

```@example ad
using AbstractPPL
using AbstractPPL: prepare, value_and_gradient, value_and_jacobian, test_autograd
using ForwardDiff  # loads AbstractPPLForwardDiffExt
using ADTypes: AutoForwardDiff

# 1. Define your problem as any callable.
struct MyModel end
(::MyModel)(x::AbstractVector) = sum(abs2, x)

model = MyModel()

# 2. Prepare: bind the model to a sample input and select an AD backend.
x0 = zeros(3)
prepared = prepare(AutoForwardDiff(), model, x0)

# 3. Evaluate and differentiate.
x = [1.0, 2.0, 3.0]
val, grad = value_and_gradient(prepared, x)
val, grad
```

## Two input styles

### Vector inputs

When the callable accepts a flat vector, pass a sample vector whose length
matches the expected input:

```@example ad
prepared_vec = prepare(AutoForwardDiff(), model, zeros(3))
value_and_gradient(prepared_vec, [1.0, 2.0, 3.0])
```

### NamedTuple inputs

When the callable accepts a `NamedTuple`, pass a sample `NamedTuple` whose
field names and value types match the expected input. The returned gradient
has the same field names and array shapes as the input:

```@example ad
struct NTModel end
function AbstractPPL.prepare(::NTModel, values::NamedTuple)
    return (values::NamedTuple) -> values.a^2 + sum(abs2, values.b)
end

nt_model = NTModel()
nt0 = (a=0.0, b=zeros(2))
prepared_nt = prepare(AutoForwardDiff(), nt_model, nt0)

val_nt, grad_nt = value_and_gradient(prepared_nt, (a=1.0, b=[2.0, 3.0]))
val_nt, grad_nt
```

## Jacobians

For vector-valued callables, use `value_and_jacobian`. The returned Jacobian
has shape `(length(value), length(x))`:

```@example ad
struct VecModel end
(::VecModel)(x::AbstractVector) = [x[1] * x[2], x[2] + x[3]]

prepared_jac = prepare(AutoForwardDiff(), VecModel(), zeros(3))
val, jac = value_and_jacobian(prepared_jac, [2.0, 3.0, 4.0])
val, jac
```

## Without an AD backend

The two-argument form `prepare(problem, x)` is available without any AD
package. It returns the callable unchanged, so code that calls `prepare`
unconditionally works regardless of which backends are loaded:

```@example ad
struct SimpleProblem end
(::SimpleProblem)(x::AbstractVector) = sum(x)

p = prepare(SimpleProblem(), zeros(3))
p([1.0, 2.0, 3.0])
```

## Testing AD correctness

[`test_autograd`](@ref AbstractPPL.test_autograd) compares `value_and_gradient`
against a finite-difference reference. It requires loading `FiniteDifferences`:

```julia
using FiniteDifferences  # loads AbstractPPLFiniteDifferencesExt
using AbstractPPL: test_autograd

test_autograd(prepared, x)       # vector input
test_autograd(prepared, values)  # NamedTuple input
```

An informative error is thrown if the AD gradient disagrees with the
finite-difference estimate.

## Supported backends

Each backend is loaded as a package extension when you load the corresponding
package:

| Package                    | `adtype`                                  | Notes                                                                                        |
|:-------------------------- |:----------------------------------------- |:-------------------------------------------------------------------------------------------- |
| `ForwardDiff`              | `AutoForwardDiff()`                       | Vector and NamedTuple inputs                                                                 |
| `Mooncake`                 | `AutoMooncake()`, `AutoMooncakeForward()` | Vector and NamedTuple inputs                                                                 |
| `FiniteDifferences`        | `AutoFiniteDifferences(; fdm)`            | Vector and NamedTuple inputs; also enables [`test_autograd`](@ref AbstractPPL.test_autograd) |
| `DifferentiationInterface` | any `ADTypes.AbstractADType`              | Vector inputs; catch-all for backends without a native extension                             |

## API reference

```@docs
AbstractPPL.prepare
AbstractPPL.value_and_gradient
AbstractPPL.value_and_jacobian
AbstractPPL.test_autograd
```
