# Evaluator preparation and AD

AbstractPPL provides a small interface for wrapping a log-density (or other
callable) into an object that AD backends can differentiate through. The design
separates *structural preparation* (binding a callable to an input prototype)
from *AD preparation* (selecting a backend and computing derivatives).

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

# 2. Prepare with a prototype vector and an AD backend.
x0 = zeros(3)
prepared = prepare(AutoForwardDiff(), model, x0)

# 3. Evaluate and differentiate.
x = [1.0, 2.0, 3.0]
val, grad = value_and_gradient(prepared, x)
val, grad
```

## Two preparation paths

### Vector inputs

When the model accepts a flat vector, pass a prototype vector of the same
length to fix the input dimension:

```@example ad
prepared_vec = prepare(AutoForwardDiff(), model, zeros(3))
value_and_gradient(prepared_vec, [1.0, 2.0, 3.0])
```

### NamedTuple inputs

When the model uses named fields, pass a `NamedTuple` prototype to fix the
field structure:

```@example ad
struct NTModel end
function AbstractPPL.prepare(::NTModel, values::NamedTuple)
    return (values::NamedTuple) -> values.a^2 + sum(abs2, values.b)
end

nt_model = NTModel()
prototype = (a=0.0, b=zeros(2))
prepared_nt = prepare(AutoForwardDiff(), nt_model, prototype)

val_nt, grad_nt = value_and_gradient(prepared_nt, (a=1.0, b=[2.0, 3.0]))
val_nt, grad_nt
```

The gradient has the same field names as the input.

## Jacobians

For vector-valued functions, use `mode=:jacobian`:

```@example ad
struct VecModel end
(::VecModel)(x::AbstractVector) = [x[1] * x[2], x[2] + x[3]]

prepared_jac = prepare(AutoForwardDiff(), VecModel(), zeros(3); mode=:jacobian)
val, jac = value_and_jacobian(prepared_jac, [2.0, 3.0, 4.0])
val, jac
```

## Without an AD backend

The two-argument form `prepare(problem, prototype)` is available without any AD
package. It binds the problem to a prototype but does not add differentiation:

```@example ad
struct SimpleProblem end
(::SimpleProblem)(x::AbstractVector) = sum(x)

p = prepare(SimpleProblem(), zeros(3))
p([1.0, 2.0, 3.0])
```

Any already-callable object is returned unchanged, so downstream code that
calls `prepare` unconditionally works even when no AD backend is loaded.

## Testing AD correctness

[`test_autograd`](@ref AbstractPPL.test_autograd) compares a prepared evaluator
against a finite-difference reference. It requires loading `FiniteDifferences`:

```julia
using FiniteDifferences  # loads AbstractPPLFiniteDifferencesExt
using AbstractPPL: test_autograd

test_autograd(prepared, x)           # vector path
test_autograd(prepared, values)      # NamedTuple path
```

An informative error is thrown if the AD gradient disagrees with the
finite-difference estimate.

## Supported backends

Each backend is loaded as a package extension; load the package to activate it:

| Package | `adtype` | Notes |
|---------|----------|-------|
| `ForwardDiff` | `AutoForwardDiff()` | Vector and NamedTuple inputs |
| `Mooncake` | `AutoMooncake()` / `AutoMooncakeForward()` | Vector and NamedTuple inputs |
| `Enzyme` | `AutoEnzyme()` | Vector inputs; forward and reverse mode |
| `FiniteDifferences` | `AutoFiniteDifferences(; fdm)` | Vector and NamedTuple; also enables [`test_autograd`](@ref AbstractPPL.test_autograd) |
| Any `DifferentiationInterface`-compatible backend | the corresponding `ADTypes` type | Vector inputs |

## API reference

```@docs
AbstractPPL.prepare
AbstractPPL.value_and_gradient
AbstractPPL.value_and_jacobian
AbstractPPL.test_autograd
AbstractPPL.ADProblems.dimension
```
