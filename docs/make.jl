using Documenter
using AbstractPPL
# trigger DistributionsExt loading
using Distributions, LinearAlgebra

# Doctest setup
DocMeta.setdocmeta!(AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true)

makedocs(;
    sitename="AbstractPPL",
    modules=[AbstractPPL, Base.get_extension(AbstractPPL, :AbstractPPLDistributionsExt)],
    pages=["Home" => "index.md", "API" => "api.md"],
    checkdocs=:exports,
    doctest=false,
)
