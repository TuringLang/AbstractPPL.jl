# AbstractPPL.jl

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.org/AbstractPPL.jl/stable/)
[![CI](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A lightweight package containing interfaces and associated APIs for modelling languages for probabilistic programming.

The [documentation](https://turinglang.org/AbstractPPL.jl/stable/) covers:

  - [VarNames and optics](https://turinglang.org/AbstractPPL.jl/stable/varname/): the `VarName` type, used throughout the TuringLang ecosystem to represent names of random variables.
  - [The `of` type system](https://turinglang.org/AbstractPPL.jl/stable/of/): a declarative, framework-agnostic way to specify parameter types for probabilistic programming.
  - [Probabilistic programming API](https://turinglang.org/AbstractPPL.jl/stable/pplapi/): the abstract model functions and trace types that downstream packages implement.
  - [Evaluator preparation and AD](https://turinglang.org/AbstractPPL.jl/stable/evaluators/): preparing a callable and asking the prepared evaluator for values and derivatives.

The [interface page](https://turinglang.org/AbstractPPL.jl/stable/interface/) records the original design discussion for a common probabilistic programming interface.
That page is marked outdated, and downstream packages are free to implement the interfaces in any appropriate way.
