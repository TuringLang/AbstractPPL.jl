using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Enzyme
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticPrepared()
end

function (::QuadraticPrepared)(x::AbstractVector{<:AbstractFloat})
    return sum(xi -> xi^2, x)
end

function AbstractPPL.ADProblems.prepare_for_test_autograd(prepared, x::AbstractVector)
    fdm = FiniteDifferences.central_fdm(5, 1)
    prepared isa typeof(AbstractPPL.prepare(ADTypes.AutoEnzyme(), QuadraticProblem(), x)) ||
        return invoke(
            AbstractPPL.ADProblems.prepare_for_test_autograd, Tuple{Any,Any}, prepared, x
        )
    return (QuadraticProblem(), x, fdm)
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

        @test prepared.mode isa Enzyme.ReverseMode
    end
end
