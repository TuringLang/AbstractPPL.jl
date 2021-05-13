# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml?query=branch%3Amain)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AbstractPPL.jl/badge.svg?branch=main)](https://coveralls.io/github/TuringLang/AbstractPPL.jl?branch=main)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A new light-weight package to factor out interfaces and associated APIs for modelling languages for
probabilistic programming.  High level goals are:

- Definition of an interface of few abstract types and a small set of functions that will be
supported all model and trace types.
- Provision of some commonly used functionality and data structures, e.g., for managing variable names and
  traces.
  
This should facilitate reuse of functions in modelling languages, to allow end users to handle
models in a consistent way, and to simplify interaction between different languages and sampler
implementations, from very rich, dynamic languages like Turing.jl to highly constrained or
simplified models such as GPs, GLMs, or plain log-density problems.

A more short term goal is to start a process of cleanly refactoring and justifying parts of
AbstractPPL.jl’s design, and hopefully to get on closer terms with Soss.jl.
