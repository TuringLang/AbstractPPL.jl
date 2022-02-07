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

@testset "AbstractPPL.jl" begin
    include("deprecations.jl")
    include("graphinfo.jl")
    @testset "doctests" begin
        DocMeta.setdocmeta!(
            AbstractPPL,
            :DocTestSetup,
            :(using AbstractPPL);
            recursive=true,
        )
        doctest(AbstractPPL; manual=false)
    end
end