# Activate test environment on older Julia versions
if VERSION < v"1.2"
    using Pkg: Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(Pkg.PackageSpec(; path=dirname(@__DIR__)))
    Pkg.instantiate()
end

using AbstractPPL
using Documenter
using Test

const GROUP = get(ENV, "GROUP", "All")

@testset "AbstractPPL.jl" begin
    if GROUP == "All" || GROUP == "Tests"
        include("deprecations.jl")
        include("varname.jl")
        include("abstractprobprog.jl")
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
