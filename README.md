# AbstractPPL.jl

[![CI](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![IntegrationTest](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml/badge.svg?branch=main)](https://github.com/TuringLang/AbstractPPL.jl/actions/workflows/IntegrationTest.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/TuringLang/AbstractPPL.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/AbstractPPL.jl)

A light-weight package to factor out interfaces and associated APIs for modelling languages for
probabilistic programming.  High level goals are:

  - Definition of an interface of few abstract types and a small set of functions that should be supported by all [probabilistic programs](./src/abstractprobprog.jl) and [trace types](./src/abstractmodeltrace.jl).
  - Provision of some commonly used functionality and data structures, e.g., for managing [variable names](./src/varname/varname.jl).

The interfaces do not currently have any specification, so downstream packages are free to implement these in any appropriate way.
Please see the documentation for more information on the design goals.
