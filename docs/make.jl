using Documenter
using AbstractPPL
# trigger DistributionsExt loading
using Distributions, LinearAlgebra
# trigger AD extension loading for adproblems.md examples
using ForwardDiff

# Doctest setup
DocMeta.setdocmeta!(AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true)

makedocs(;
    sitename="AbstractPPL",
    modules=[
        AbstractPPL,
        Base.get_extension(AbstractPPL, :AbstractPPLDistributionsExt),
        Base.get_extension(AbstractPPL, :AbstractPPLForwardDiffExt),
    ],
    pages=["index.md", "varname.md", "pplapi.md", "adproblems.md", "interface.md"],
    checkdocs=:exports,
    doctest=false,
)
