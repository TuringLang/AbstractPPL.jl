using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Enzyme
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

@testset "AbstractPPLEnzymeExt" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(ADTypes.AutoEnzyme(), x0, x)
    run_shared_jacobian_tests(
        ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)),
        x0,
        [2.0, 3.0, 4.0],
    )
    run_shared_jacobian_tests(
        ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
        x0,
        [2.0, 3.0, 4.0],
    )

    @testset "honors AutoEnzyme mode" begin
        fwd = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward))
        rev = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
        problem = QuadraticProblem()
        prepared_fwd = AbstractPPL.prepare(fwd, problem, x0)
        prepared_rev = AbstractPPL.prepare(rev, problem, x0)

        @test prepared_fwd.mode isa Enzyme.ForwardMode
        @test prepared_rev.mode isa Enzyme.ReverseMode
        @test typeof(prepared_fwd) !== typeof(prepared_rev)

        val_fwd, grad_fwd = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
            prepared_fwd, x
        )
        @test val_fwd ≈ 14.0
        @test grad_fwd ≈ [6.0, 2.0, 4.0]
    end

    @testset "normalizes single-parameter forward gradients" begin
        fwd = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward))
        x1 = [3.0]
        prepared_fwd = AbstractPPL.prepare(fwd, QuadraticProblem(), zeros(1))

        val_fwd, grad_fwd = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
            prepared_fwd, x1
        )
        @test val_fwd ≈ 9.0
        @test grad_fwd ≈ [6.0]
    end
end
