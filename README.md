# AbstractPPL.jl

A new light-weight package, refactoring of `AbstractVarInfo` and associated APIs. The overall goals
are creating an abstract type and minimal set of functions that will be supported all `VarInfo`
types, e.g.

- `TypedVarInfo`: VarInfo with typed model parameters, mostly for HMC,
- `UntypedVarInfo`: VarInfo with untyped model parameters,
- `MixedVarInfo`: VarInfo with both typed and untyped parameters. 

This will likely take some time, but the hope is to unify VarInfo, improving its clarity and
modularity.  The package will also host other future work like `NoVarInfo` for potential integration
with non-tracing PPLs, `GraphVarInfo` for BUGS style models with statically known dependency
structure, and generic modelling combinators (e.g. For, Switch, IID ) that are common for all PPLs.

Not related to https://github.com/JuliaPPL/AbstractPPL.jl.
