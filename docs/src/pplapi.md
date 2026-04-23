# Probabilistic programming API

## Abstract model functions

```@docs
AbstractProbabilisticProgram
condition
decondition
fix
unfix
logdensityof
AbstractContext
evaluate!!
```

## Abstract traces

```@docs
AbstractModelTrace
```

## ADProblem interface

```@docs
DerivativeCapability
capabilities
prepare
value_and_gradient
value_and_jacobian
test_autograd
dimension
```
