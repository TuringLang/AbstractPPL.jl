using Documenter
using AbstractPPL
using Random  # for the `Base.rand(::Random.AbstractRNG, ...)` signature in of.md's @docs block
# trigger DistributionsExt loading
using Distributions, LinearAlgebra

# Doctest setup
DocMeta.setdocmeta!(AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true)

makedocs(;
    sitename="AbstractPPL",
    modules=[AbstractPPL, Base.get_extension(AbstractPPL, :AbstractPPLDistributionsExt)],
    pages=[
        "index.md",
        "varname.md",
        "varnamedtuple.md",
        "of.md",
        "pplapi.md",
        "evaluators.md",
        "interface.md",
    ],
    checkdocs=:exports,
    doctest=false,
)
