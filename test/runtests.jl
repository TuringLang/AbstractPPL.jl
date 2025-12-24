using AbstractPPL
using Documenter
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "AbstractPPL.jl" begin
    if GROUP == "All" || GROUP == "Tests"
        # include("Aqua.jl")
        include("abstractprobprog.jl")
        include("varname/optic.jl")
        include("varname/varname.jl")
        include("varname/subsumes.jl")
        include("varname/hasvalue.jl")
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
