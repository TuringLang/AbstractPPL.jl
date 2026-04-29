using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: AutoForwardDiff
using ForwardDiff
using Test

include(joinpath(@__DIR__, "..", "..", "autograd_tests.jl"))

@testset "AbstractPPLForwardDiffExt" begin
    run_autograd_tests(AutoForwardDiff(); namedtuple=true)
end
