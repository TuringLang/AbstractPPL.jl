# Suggestion: make evaluator shapes first-class in `ADProblems`

The high-level goal is to make `AbstractPPL.ADProblems` a complete downstream-facing abstraction for AD-backed evaluators, so downstream packages only need to describe *what their evaluator is* and *what input shape it accepts*, while AbstractPPL owns backend preparation and execution.

## Problem

The current API is conceptually clean:

  - `prepare(problem, x)` / `prepare(problem, values)`
  - `prepare(adtype, problem, x)` / backend package extensions
  - `capabilities`
  - `dimension`
  - `value_and_gradient`

But in practice, downstream packages still need backend-specific glue when their evaluator wrapper does not already match the assumptions of each backend extension.

That means the abstraction boundary is still somewhat leaky: backend logic lives in AbstractPPL, but downstream packages may still need to write one adapter per backend.

## Suggestion

Make evaluator *shapes* first-class in `ADProblems`, and have backend extensions consume those shapes uniformly.

There should be two equally supported evaluator families:

 1. **Vector evaluators**
    
      + callable as `f(x::AbstractVector{<:AbstractFloat}) -> Real`
      + expose `AbstractPPL.dimension(f)::Int`

 2. **NamedTuple evaluators**
    
      + callable as `f(values::NamedTuple) -> Real`
      + expose a stable input schema/prototype used for preparation and reconstruction

## Desired downstream contract

A downstream package should be able to provide either:

### Vector-input evaluator

An evaluator `f` such that:

  - `f(x)` returns the scalar value
  - `AbstractPPL.dimension(f)` is defined
  - optionally `AbstractPPL.capabilities(typeof(f)) = DerivativeOrder{0}()`

and then expect this to be enough for supported backends like:

  - `prepare(AutoForwardDiff(), f, x)`
  - `prepare(AutoMooncake(), f, x)`
  - `prepare(AutoMooncakeForward(), f, x)`
  - other vector AD backends

provided the backend supports ordinary vector-input scalar-output functions.

### NamedTuple-input evaluator

An evaluator `f` such that:

  - `f(values::NamedTuple)` returns the scalar value
  - preparation is given a stable `NamedTuple` prototype/schema
  - backend extensions may flatten/unflatten internally as needed

and then expect backend AD support without writing backend-specific downstream adapters.

## The promise AbstractPPL should make

In effect, AbstractPPL should be able to say:

  - “If you give me a vector evaluator plus `dimension`, I can prepare it for supported AD backends.”
  - “If you give me a NamedTuple evaluator plus a stable schema/prototype, I can prepare it for supported AD backends.”

That would give downstream packages a self-contained and backend-agnostic integration story.

## Division of responsibility

This would create a clean boundary.

### Downstream packages own

  - turning models / prepared state into evaluators
  - defining the input shape
  - defining `dimension` for vector evaluators, or supplying a stable schema/prototype for NamedTuple evaluators
  - defining plain scalar evaluation

### AbstractPPL owns

  - backend preparation
  - flattening/unflattening when needed
  - backend-specific execution of `value_and_gradient`
  - test helpers like `prepare_for_test_autograd`

## Why this is useful

This would let downstream packages like DynamicPPL stop at the natural abstraction boundary:

  - convert model state into a vector-callable or NamedTuple-callable evaluator
  - define the required shape metadata
  - rely on AbstractPPL from there

That is much simpler than requiring per-backend adapter methods for otherwise sensible evaluator wrappers.

## Minimal design direction

There are two reasonable ways to realize this.

### Option 1: canonical wrapper types

Define explicit wrapper types in AbstractPPL, e.g.:

  - `VectorEvaluator(f, dim)`
  - `NamedTupleEvaluator(f, prototype)`

Then backend extensions can target those wrappers directly.

### Option 2: structural interface

Document and support a structural contract:

#### For vector evaluators

  - callable on vector input
  - `dimension(::Evaluator)`

#### For NamedTuple evaluators

  - callable on `NamedTuple` input
  - stable input schema/prototype available at preparation time

Then require all backend extensions to consume those contracts uniformly.

The structural approach is more flexible, while wrapper types may make backend implementations simpler and more explicit.

## Recommendation

I would recommend making these evaluator shapes explicit and first-class in `ADProblems`, whether through wrapper types or a clearly documented structural contract.

The important part is not the exact surface syntax, but the guarantee to downstream packages:

  - vector-callable evaluator + `dimension` is enough
  - NamedTuple-callable evaluator + stable schema/prototype is enough
  - backend-specific AD preparation remains in AbstractPPL, not in downstream packages

That would make the evaluator API not just conceptually sufficient, but operationally sufficient as a downstream abstraction.
