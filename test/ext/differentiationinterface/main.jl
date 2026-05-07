using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: run_testcases
using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using Test

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
    run_testcases(Val(:edge); adtype=AutoForwardDiff())
end
