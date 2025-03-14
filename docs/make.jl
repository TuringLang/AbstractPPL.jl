using Documenter
using AbstractPPL

# Doctest setup
DocMeta.setdocmeta!(AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true)

makedocs(;
    sitename="AbstractPPL",
    modules=[AbstractPPL],
    pages=["Home" => "index.md", "API" => "api.md"],
    checkdocs=:exports,
    doctest=false,
)
