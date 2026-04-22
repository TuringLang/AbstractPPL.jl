using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Enzyme
using FiniteDifferences
using Test

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(x::AbstractVector{<:AbstractFloat})
    return sum(xi -> xi^2, x)
end

@testset "AbstractPPLEnzymeExt" begin
    problem = QuadraticProblem()
    x0 = zeros(3)
    prepared = AbstractPPL.prepare(ADTypes.AutoEnzyme(), problem, x0)

    @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
    @test AbstractPPL.dimension(prepared) == 3

    x = [3.0, 1.0, 2.0]
    @test prepared(x) ≈ 14.0

    val, grad = AbstractPPL.value_and_gradient(prepared, x)
    @test val ≈ 14.0
    @test grad ≈ [6.0, 2.0, 4.0]
    test_autograd(prepared, x)

    @test_throws DimensionMismatch prepared([3.0, 1.0, 2.0, 99.0])
    @test_throws MethodError prepared([3, 1, 2])

    @testset "honors AutoEnzyme mode" begin
        fwd = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Forward))
        rev = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
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
        prepared_fwd = AbstractPPL.prepare(fwd, problem, zeros(1))

        val_fwd, grad_fwd = @inferred Tuple{Float64,Vector{Float64}} AbstractPPL.value_and_gradient(
            prepared_fwd, x1
        )
        @test val_fwd ≈ 9.0
        @test grad_fwd ≈ [6.0]
    end
end
