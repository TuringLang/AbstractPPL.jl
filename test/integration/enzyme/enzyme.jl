using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using Enzyme
using Test

include(joinpath(@__DIR__, "..", "..", "ext", "ad_tests.jl"))

struct StatefulQuadraticProblem
    data::Vector{Float64}
end

function AbstractPPL.prepare(problem::StatefulQuadraticProblem, x::AbstractVector{<:Real})
    return problem
end

function (p::StatefulQuadraticProblem)(x::AbstractVector{<:Real})
    return sum(abs2(xi - di) for (xi, di) in zip(x, p.data))
end

@testset "Enzyme via DifferentiationInterface" begin
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

        val_fwd, grad_fwd = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
            prepared_fwd, x
        )
        @test val_fwd ≈ 14.0
        @test grad_fwd ≈ [6.0, 2.0, 4.0]

        val_rev, grad_rev = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
            prepared_rev, x
        )
        @test val_rev ≈ 14.0
        @test grad_rev ≈ [6.0, 2.0, 4.0]
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

    prepared = AbstractPPL.prepare(
        ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward)),
        StatefulQuadraticProblem([0.5, 1.5]),
        zeros(2),
    )
    val, grad = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
        prepared, [1.0, 2.0]
    )
    @test val ≈ 0.5
    @test grad ≈ [1.0, 1.0]
end
