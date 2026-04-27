using AbstractPPL
using AbstractPPL: prepare, value_and_gradient, value_and_jacobian
using ADTypes: ADTypes
using Test

struct DummyProblem end

struct DummyPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(problem::DummyProblem, values::NamedTuple)
    return DummyPrepared(keys(values))
end

function (p::DummyPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys ||
        error("expected fields $(p.prototype_keys), got $(keys(values))")
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

struct DummyADPrepared
    dim::Int
end

struct DummyADType <: ADTypes.AbstractADType end

function AbstractPPL.prepare(
    ::DummyADType, problem::DummyProblem, x::AbstractVector{<:Real}
)
    return DummyADPrepared(length(x))
end

function (p::DummyADPrepared)(x::AbstractVector{<:Real})
    length(x) == p.dim || error("expected vector of length $(p.dim)")
    return sum(x)
end

function AbstractPPL.value_and_gradient(p::DummyADPrepared, x::AbstractVector{<:Real})
    return (sum(x), ones(length(x)))
end

struct ZeroDimProblem end
struct ZeroDimVecProblem end

function AbstractPPL.prepare(::ZeroDimProblem, ::AbstractVector{<:Real})
    return (_::AbstractVector) -> 7.5
end
function AbstractPPL.prepare(::ZeroDimVecProblem, ::AbstractVector{<:Real})
    return (_::AbstractVector) -> [2.0, 3.0]
end

@testset "ADProblem interface" begin
    @testset "explicit evaluator shapes" begin
        ve = AbstractPPL.ADProblems.VectorEvaluator(sum, 3)
        @test ve([1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch ve([1.0, 2.0])
        @test_throws MethodError ve([1, 2, 3])

        ne = AbstractPPL.ADProblems.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test ne((a=1.0, b=[2.0, 3.0])) == 6.0
        @test ne.inputspec == (a=0.0, b=zeros(2))
        @test_throws MethodError ne([1.0, 2.0, 3.0])

        # `Validate=false` skips the per-call shape checks.
        ve_unchecked = AbstractPPL.ADProblems.VectorEvaluator{false}(sum, 3)
        @test ve_unchecked([1.0, 2.0]) == 3.0

        ne_unchecked = AbstractPPL.ADProblems.NamedTupleEvaluator{false}(
            x -> 0.0, (a=0.0, b=zeros(2))
        )
        @test AbstractPPL.ADProblems._assert_namedtuple_shape(
            ne_unchecked, (totally=:wrong,)
        ) === nothing
        @test_throws r"same NamedTuple structure" AbstractPPL.ADProblems._assert_namedtuple_shape(
            ne, (totally=:wrong,)
        )
    end

    @testset "prepare (structural)" begin
        problem = DummyProblem()
        values = (x=0.0, y=[1.0, 2.0])
        prepared = prepare(problem, values)
        @test prepared isa DummyPrepared
        @test prepared.prototype_keys == (:x, :y)

        lp = prepared((x=0.5, y=[1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        @test_throws ErrorException prepared((a=1.0, b=2.0))
    end

    @testset "prepare (AD-aware)" begin
        problem = DummyProblem()
        x0 = zeros(3)
        adtype = DummyADType()
        prepared = prepare(adtype, problem, x0)
        @test prepared isa DummyADPrepared

        x = [0.5, 1.5, 2.5]
        @test prepared(x) ≈ 0.5 + 1.5 + 2.5

        val, grad = value_and_gradient(prepared, x)
        @test val ≈ 0.5 + 1.5 + 2.5
        @test grad ≈ [1.0, 1.0, 1.0]
    end

    @testset "missing AD package extensions" begin
        problem = DummyProblem()
        x0 = zeros(3)

        @test_throws MethodError AbstractPPL.ADProblems.prepare(
            ADTypes.AutoEnzyme(), problem, x0
        )
    end

    @testset "flatten / unflatten edge cases" begin
        empty = NamedTuple()
        @test AbstractPPL.Utils.flatten_to!!(nothing, empty) == Float64[]
        @test AbstractPPL.Utils.unflatten_to!!(empty, Float64[]) == empty

        view_values = (x=@view([1.0, 2.0, 3.0][2:3]),)
        flat = AbstractPPL.Utils.flatten_to!!(nothing, view_values)
        rebuilt = AbstractPPL.Utils.unflatten_to!!(view_values, flat)
        @test collect(rebuilt.x) == [2.0, 3.0]
    end

    @testset "zero-dimensional prepared evaluator" begin
        x = Float64[]

        prepared = AbstractPPL.ADProblems.VectorEvaluator{true}(
            AbstractPPL.prepare(ZeroDimProblem(), x), 0
        )
        val, grad = value_and_gradient(prepared, x)
        @test val == 7.5
        @test grad == Float64[]

        prepared_jac = AbstractPPL.ADProblems.VectorEvaluator{true}(
            AbstractPPL.prepare(ZeroDimVecProblem(), x), 0
        )
        valj, jac = value_and_jacobian(prepared_jac, x)
        @test valj == [2.0, 3.0]
        @test size(jac) == (2, 0)
    end
end
