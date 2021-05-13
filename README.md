# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/workflows/IntegrationTest/badge.svg?branch=master)](https://github.com/TuringLang/AbstractPPL.jl/actions?query=workflow%3AIntegrationTest+branch%3Amaster)
[![Coverage Status](https://coveralls.io/repos/github/TuringLang/AbstractPPL.jl/badge.svg?branch=master)](https://coveralls.io/github/TuringLang/AbstractPPL.jl?branch=master)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

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

See [interface draft](interface.md).
