using AbstractPPL
using Documenter
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "AbstractPPL.jl" begin
    if GROUP == "All" || GROUP == "Tests"
        include("Aqua.jl")
        include("abstractprobprog.jl")
        include("evaluators/Evaluators.jl")
        include("evaluators/utils.jl")
        include("varname/optic.jl")
        include("varname/varname.jl")
        include("varname/subsumes.jl")
        include("varname/hasvalue.jl")
        include("varname/leaves.jl")
        include("varname/serialize.jl")
        include("varnamedtuple.jl")
        include("of.jl")
    end

    if GROUP == "All" || GROUP == "Doctests"
        @testset "doctests" begin
            DocMeta.setdocmeta!(
                AbstractPPL, :DocTestSetup, :(using AbstractPPL); recursive=true
            )
            doctestfilters = [r"└ @ .+:[0-9]+"]
            doctest(AbstractPPL; manual=false, doctestfilters)
        end
    end
end
