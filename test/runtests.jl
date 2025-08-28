using AbstractPPL
using Documenter
using Test

const GROUP = get(ENV, "GROUP", "All")
const AQUA = get(ENV, "AQUA", "true") == "true"

@testset "AbstractPPL.jl" begin
    if GROUP == "All" || GROUP == "Tests"
        if AQUA
            include("Aqua.jl")
        end
        include("varname.jl")
        include("abstractprobprog.jl")
        include("hasvalue.jl")
    end

    if GROUP == "All" || GROUP == "Doctests"
        @testset "doctests" begin
            DocMeta.setdocmeta!(
                AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true
            )
            doctest(AbstractPPL; manual=false)
        end
    end
end
