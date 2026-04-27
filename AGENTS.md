# AGENTS.md

AbstractPPL.jl is a Julia interface package for probabilistic programming. It is used by `DynamicPPL.jl` (`https://github.com/TuringLang/DynamicPPL.jl`) and `JuliaBUGS.jl` (`https://github.com/TuringLang/JuliaBUGS.jl`). Most of the package is contract surface: a small model API, a prepared-evaluator API, and a nontrivial `VarName`/optic subsystem used across the Turing ecosystem.

## Conventions

  - Make the smallest correct change.
  - Treat exported behaviour, examples, and tests as the package contract.
  - Keep source, tests, and docs aligned for public behaviour changes.
  - Prefer narrow, explicit methods over broad signatures.
  - Keep weakdep-specific behaviour in `ext/`.
  - Prefer targeted fixes over broad refactors.
  - Avoid new dependencies unless clearly justified.

## Package-Specific Notes

  - Preserve the model invariants documented in `src/abstractprobprog.jl`: `condition`/`decondition` and `fix`/`unfix` are intended to round-trip when supported.
  - `rand(model)` and `predict(model, params)` have default RNG/type forwarding behaviour covered by tests; changes here should stay consistent with `AbstractMCMC` expectations.
  - The ADProblem API in `src/ADProblems.jl` is structural. `prepare(..., prototype::NamedTuple)` fixes field structure, `capabilities` defaults conservatively to `DerivativeOrder{0}()`, and AD-aware prepared objects are expected to return gradients with the same named structure as inputs.
  - `VarName` and optics are the main complexity in this repo. Preserve equality, hashing, pretty-printing, composition/decomposition, and type-stability behaviour.
  - Dynamic indices (`begin`, `end`, expressions containing them) are intentionally deferred until `concretize`; do not silently erase that distinction.
  - Unconcretized dynamic indices must not be serialised. If serialization changes, keep `varname_to_string` / `string_to_varname` round-tripping for supported index types.
  - `@varname(..., true)` cannot auto-concretize when the top-level symbol is interpolated; preserve that error path unless the design is intentionally changed.
  - `subsumes` is conservative by design. Do not broaden it casually for ambiguous indexing forms.
  - `getvalue` / `hasvalue` on `AbstractDict{<:VarName}` intentionally prefer exact matches, then walk up to more general parents when possible.
  - The Distributions extension exists to reconstruct structured values from elementwise `VarName` entries. Keep that logic in `ext/`, and prefer the simpler two-argument `getvalue` / `hasvalue` methods unless distribution-shaped reconstruction is actually needed.
  - There is explicit code to keep optic equality JET-friendly and to work around Julia tuple-equality issues; changes around optic equality need extra care.

## Julia Patterns

  - Prefer `Base.Experimental.register_error_hint` in `__init__` over catch-all throwing methods for helpful `MethodError` messages. Catch-all methods create method ambiguities; error hints do not. Note that hint text is shown at display time only — it is not stored in the exception, so `@test_throws r"..."` will not match it.
  - Prefer two-method dispatch over a single method with an `isa` check: `_assert_foo(_) = nothing` / `_assert_foo(::T) = throw(...)` is cleaner, more composable, and avoids value-level branching.

## Testing

  - Start with a minimal reproducer and run the smallest relevant test scope.
  - Main test command: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - CI-style variants:
    `GROUP=Tests julia --project=. -e 'using Pkg; Pkg.test()'`
    `GROUP=Doctests julia --project=. -e 'using Pkg; Pkg.test()'`
  - VarName or optic changes usually need coverage for equality/hash, concretization, `hasvalue`/`getvalue`, serialization, and JET-sensitive cases.
  - Public API changes should keep Aqua and doctests passing.
  - If a change may affect ecosystem compatibility, consider the downstream `DynamicPPL.jl` integration workflow as part of validation.
  - Do not weaken tests just to make CI pass without explicit confirmation.

## Docs

  - Keep `docs/src/` aligned with public API and examples.
  - `docs/src/interface.md` is explicitly marked outdated and aspirational; prefer current source, tests, and docstrings over that page when they conflict.
  - Build docs with: `julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'`
  - Follow `.JuliaFormatter.toml` when formatting is part of the task.
