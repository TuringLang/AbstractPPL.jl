using Documenter
using AbstractPPL
# trigger DistributionsExt loading
using Distributions, LinearAlgebra

# Doctest setup
DocMeta.setdocmeta!(AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true)

makedocs(;
    sitename="AbstractPPL",
    modules=[AbstractPPL, Base.get_extension(AbstractPPL, :AbstractPPLDistributionsExt)],
    pages=["index.md", "api.md", "interface.md"],
    checkdocs=:exports,
    doctest=false,
)
