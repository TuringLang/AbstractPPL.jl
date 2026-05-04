using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using AbstractPPL.TestResources: generate_testcases
using ADTypes: ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Test

# Stub backend without a native AbstractPPL extension; this exercises the
# `AbstractPPLDifferentiationInterfaceExt` catch-all dispatch on
# `<:AbstractADType` rather than depending on a real AD package.
struct DummyADType <: ADTypes.AbstractADType end
const adtype = DummyADType()

DI.prepare_gradient(f, ::DummyADType, x, ::DI.Constant) = Val(:gradient)
function DI.value_and_gradient(f, prep, ::DummyADType, x, ctx::DI.Constant)
    return (f(x, ctx.data), 2 .* x)
end
DI.prepare_jacobian(f, ::DummyADType, x, ::DI.Constant) = Val(:jacobian)
function DI.value_and_jacobian(f, prep, ::DummyADType, x, ctx::DI.Constant)
    jac = [
        x[2] x[1] zero(eltype(x))
        zero(eltype(x)) one(eltype(x)) one(eltype(x))
    ]
    return (f(x, ctx.data), jac)
end

const ATOL = 1e-6
const RTOL = 1e-6

@testset "AbstractPPLDifferentiationInterfaceExt" begin
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
