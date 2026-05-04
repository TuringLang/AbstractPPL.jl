using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using AbstractPPL.TestResources: generate_testcases
using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using Test

const adtype = AutoForwardDiff()
const ATOL = 1e-6
const RTOL = 1e-6

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    # Use a real DI backend to exercise AbstractPPL's catch-all ADType dispatch.
    @testset "ForwardDiff" begin
        @testset "vector input" begin
            @testset "$(case.name)" for case in generate_testcases(Val(:vector))
                prepared = AbstractPPL.prepare(adtype, case.f, case.x_proto)
                @test prepared(case.x) ≈ case.value atol = ATOL rtol = RTOL
                if case.gradient !== nothing
                    val, grad = AbstractPPL.value_and_gradient!!(prepared, case.x)
                    @test val ≈ case.value atol = ATOL rtol = RTOL
                    @test grad ≈ case.gradient atol = ATOL rtol = RTOL
                end
                if case.jacobian !== nothing
                    val, jac = AbstractPPL.value_and_jacobian!!(prepared, case.x)
                    @test val ≈ case.value atol = ATOL rtol = RTOL
                    @test jac ≈ case.jacobian atol = ATOL rtol = RTOL
                end
            end

            @testset "edge cases" begin
                @testset "$(case.name)" for case in generate_testcases(Val(:edge))
                    prepared = AbstractPPL.prepare(adtype, case.f, case.x_proto)
                    if case.operation === :call
                        @test_throws case.exception prepared(case.x)
                    elseif case.operation === :gradient
                        @test_throws case.exception AbstractPPL.value_and_gradient!!(
                            prepared, case.x
                        )
                    else
                        error("Unknown edge-test operation: $(case.operation)")
                    end
                end
            end
        end
    end
end
