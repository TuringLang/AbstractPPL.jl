using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Mooncake
using Test

include(joinpath(@__DIR__, "..", "..", "test_utils.jl"))

struct QuadraticProblem end
struct QuadraticNTPrepared end
struct QuadraticVecPrepared end

function AbstractPPL.prepare(::QuadraticProblem, values::NamedTuple)
    return QuadraticNTPrepared()
end

function (::QuadraticNTPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

function AbstractPPL.prepare(::QuadraticProblem, x::AbstractVector{<:AbstractFloat})
    return QuadraticVecPrepared()
end

function (::QuadraticVecPrepared)(x::AbstractVector{<:AbstractFloat})
    return sum(xi -> xi^2, x)
end

const config = ADTypes.AutoMooncake(; config=Mooncake.Config())

@testset "AbstractPPLMooncakeExt" begin
    @testset "NamedTuple path" begin
        problem = QuadraticProblem()
        values = (x=0.0, y=[0.0, 0.0])
        prepared = AbstractPPL.prepare(config, problem, values)

        @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
        err = try
            AbstractPPL.dimension(prepared)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin(
            "only available for evaluators prepared with a vector", sprint(showerror, err)
        )

        values = (x=3.0, y=[1.0, 2.0])
        @test prepared(values) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, values)
        @test val ≈ 14.0
        @test grad.x ≈ 6.0
        @test grad.y ≈ [2.0, 4.0]

        @test_throws MethodError prepared((x=3.0, z=[1.0, 2.0]))
        err = try
            AbstractPPL.value_and_gradient(prepared, (x=3.0, y=reshape([1.0, 2.0], 1, 2)))
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin("same NamedTuple structure", sprint(showerror, err))
    end

    @testset "vector path" begin
        problem = QuadraticProblem()
        x0 = zeros(3)
        prepared = AbstractPPL.prepare(config, problem, x0)

        @test AbstractPPL.capabilities(prepared) >= AbstractPPL.DerivativeOrder{1}()
        @test AbstractPPL.dimension(prepared) == 3

        x = [3.0, 1.0, 2.0]
        @test prepared(x) ≈ 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, x)
        @test val ≈ 14.0
        @test grad ≈ [6.0, 2.0, 4.0]
        test_autograd(prepared, x)

        err = try
            prepared([3.0, 1.0, 2.0, 99.0])
            nothing
        catch err
            err
        end
        @test err isa DimensionMismatch
        @test occursin(
            "Expected a vector of length 3, but got length 4.", sprint(showerror, err)
        )
        @test_throws MethodError prepared([3, 1, 2])
        @test_throws DimensionMismatch AbstractPPL.value_and_gradient(
            prepared, [3.0, 1.0, 2.0, 3.0]
        )
    end
end
