# AGENTS.md

## Purpose

AbstractPPL.jl provides shared interfaces and utilities for probabilistic programming packages. Keep it small, conservative, and downstream-friendly: public contracts matter more than local convenience.

## Layout

  - `src/AbstractPPL.jl`: exports, includes, and `public` declarations
  - `src/abstractprobprog.jl`: `AbstractProbabilisticProgram`, conditioning, fixing, density, sampling, and prediction hooks
  - `src/evaluators/`: `prepare`, `Prepared`, `VectorEvaluator`, `NamedTupleEvaluator`, and flattening helpers for AD backends
  - `src/varname/`: `VarName`, optics, prefix/subsumption, leaves, value lookup, and serialization
  - `ext/`: weak dependency integrations
  - `test/`: core tests, doctests, and extension launchers
  - `docs/src/`: documentation pages wired by `docs/make.jl`

## Working Rules

  - Treat this as interface infrastructure. Avoid broad fallbacks or feature-heavy implementations that make unsupported model operations appear to work.
  - Preserve exported and public behaviour unless intentionally making a breaking change; update tests and docs with any API change.
  - Put optional dependency integrations in `[weakdeps]`, `[extensions]`, `ext/`, and isolated tests when behaviour is nontrivial.
  - Keep model-operation semantics aligned with `condition`, `decondition`, `fix`, and `unfix` invariants in `src/abstractprobprog.jl`.
  - Do not assume traces are `NamedTuple`s or dictionaries; downstream packages may provide richer trace types.
  - When touching `VarName` or optics code, test symbolic, indexed, nested, and serialization round-trip cases.
  - Respect evaluator contracts: `VectorEvaluator` is for flat vectors; `NamedTupleEvaluator` is for stable named structures; `!!` derivative APIs may return cache-aliased arrays.
  - Use `check_dims=false` only for trusted AD hot paths. Public evaluator calls should validate user input.

## Tests

  - Core tests: `GROUP=Tests julia --project=test test/runtests.jl`
  - Doctests: `GROUP=Doctests julia --project=test test/runtests.jl`
  - Full package tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Docs: `julia --project=docs docs/make.jl`

Run the smallest relevant test first, then broaden when changing public interfaces, extensions, or downstream-facing behaviour. Do not weaken tests just to make CI pass.

## Documentation

`docs/src/interface.md` is marked outdated and aspirational; prefer current docstrings and `docs/src/evaluators.md` for evaluator and AD contracts. Keep `docs/make.jl` navigation in sync with new pages.
