using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using FiniteDifferences
using ForwardDiff
using Test

struct QuadraticProblem end
struct QuadraticNTPrepared end
struct QuadraticVecPrepared end
struct RequestedTag end

function AbstractPPL.prepare(::QuadraticProblem, values::NamedTuple)
    return QuadraticNTPrepared()
end

function (::QuadraticNTPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticVecPrepared()
end

function (::QuadraticVecPrepared)(x::AbstractVector{<:Real})
    return sum(xi -> xi^2, x)
end

function AbstractPPL.ADProblems.prepare_for_test_autograd(prepared, x::AbstractVector)
    fdm = FiniteDifferences.central_fdm(5, 1)
    prepared isa
    typeof(AbstractPPL.prepare(ADTypes.AutoForwardDiff(), QuadraticProblem(), x)) ||
        return invoke(
            AbstractPPL.ADProblems.prepare_for_test_autograd, Tuple{Any,Any}, prepared, x
        )
    return (QuadraticProblem(), x, fdm)
end

@testset "AbstractPPLForwardDiffExt" begin
    @testset "NamedTuple path" begin
        problem = QuadraticProblem()
        values = (x=0.0, y=[0.0, 0.0])
        prepared = AbstractPPL.prepare(ADTypes.AutoForwardDiff(), problem, values)

        @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
        @test_throws r"only available for evaluators prepared with a vector" AbstractPPL.dimension(
            prepared
        )

        values = (x=3.0, y=[1.0, 2.0])
        @test prepared(values) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, values)
        @test val ≈ 14.0
        @test grad.x ≈ 6.0
        @test grad.y ≈ [2.0, 4.0]

        @test_throws r"same NamedTuple structure" prepared((x=3.0, z=[1.0, 2.0]))
        @test_throws r"same NamedTuple structure" AbstractPPL.value_and_gradient(
            prepared, (x=3.0, y=reshape([1.0, 2.0], 1, 2))
        )
    end

    @testset "vector path" begin
        problem = QuadraticProblem()
        x0 = zeros(3)
        prepared = AbstractPPL.prepare(ADTypes.AutoForwardDiff(), problem, x0)

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

        @testset "check_dims=false skips dim/shape checks" begin
            prepared_unchecked = AbstractPPL.prepare(
                ADTypes.AutoForwardDiff(), problem, x0; check_dims=false
            )
            val, grad = AbstractPPL.value_and_gradient(prepared_unchecked, x)
            @test val ≈ 14.0
            @test grad ≈ [6.0, 2.0, 4.0]

            values = (x=3.0, y=[1.0, 2.0])
            prepared_nt_unchecked = AbstractPPL.prepare(
                ADTypes.AutoForwardDiff(), problem, (x=0.0, y=[0.0, 0.0]); check_dims=false
            )
            @test prepared_nt_unchecked(values) ≈ 14.0
        end

        @testset "honors caller-provided custom tag" begin
            ad = ADTypes.AutoForwardDiff(;
                chunksize=1, tag=ForwardDiff.Tag(RequestedTag(), Float64)
            )
            prepared_tagged = AbstractPPL.prepare(ad, problem, x0)
            val, grad = AbstractPPL.value_and_gradient(prepared_tagged, x)
            @test val ≈ 14.0
            @test grad ≈ [6.0, 2.0, 4.0]
        end
    end
end
