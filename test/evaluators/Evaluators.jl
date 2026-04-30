using AbstractPPL
using AbstractPPL: prepare, value_and_gradient!!, evaluate!!
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
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

struct DummyADType <: ADTypes.AbstractADType end

function AbstractPPL.prepare(
    adtype::DummyADType, problem::DummyProblem, x::AbstractVector{<:Real};
    check_dims::Bool=true,
)
    f = x -> sum(x)
    return Prepared(adtype, VectorEvaluator{check_dims}(f, length(x)))
end

function AbstractPPL.value_and_gradient!!(
    p::Prepared{DummyADType}, x::AbstractVector{<:Real}
)
    return (sum(x), ones(length(x)))
end

@testset "ADProblem interface" begin
    @testset "explicit evaluator shapes" begin
        ve = AbstractPPL.Evaluators.VectorEvaluator(sum, 3)
        @test ve([1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch ve([1.0, 2.0])
        @test_throws r"floating-point" ve([1, 2, 3])

        ne = AbstractPPL.Evaluators.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test ne((a=1.0, b=[2.0, 3.0])) == 6.0
        @test ne.inputspec == (a=0.0, b=zeros(2))
        @test_throws MethodError ne([1.0, 2.0, 3.0])

        # `CheckInput=false` skips the per-call shape checks.
        ve_unchecked = AbstractPPL.Evaluators.VectorEvaluator{false}(sum, 3)
        @test ve_unchecked([1.0, 2.0]) == 3.0

        ne_unchecked = AbstractPPL.Evaluators.NamedTupleEvaluator{false}(
            x -> 0.0, (a=0.0, b=zeros(2))
        )
        @test AbstractPPL.Evaluators._assert_namedtuple_shape(
            ne_unchecked, (totally=:wrong,)
        ) === nothing
        @test_throws r"same NamedTuple structure" AbstractPPL.Evaluators._assert_namedtuple_shape(
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
        @test prepared isa Prepared{DummyADType}

        x = [0.5, 1.5, 2.5]
        @test prepared(x) ≈ 0.5 + 1.5 + 2.5

        val, grad = value_and_gradient!!(prepared, x)
        @test val ≈ 0.5 + 1.5 + 2.5
        @test grad ≈ [1.0, 1.0, 1.0]

        # check_dims=false skips the per-call dimension check.
        prepared_unchecked = prepare(adtype, problem, x0; check_dims=false)
        @test prepared_unchecked([1.0, 2.0]) ≈ 3.0  # wrong length, no error
    end

    @testset "missing AD package extensions" begin
        problem = DummyProblem()
        x0 = zeros(3)

        @test_throws MethodError AbstractPPL.Evaluators.prepare(
            ADTypes.AutoEnzyme(), problem, x0
        )
    end

    @testset "evaluate!!" begin
        ve = AbstractPPL.Evaluators.VectorEvaluator(sum, 3)
        @test evaluate!!(ve, [1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch evaluate!!(ve, [1.0, 2.0])

        ne = AbstractPPL.Evaluators.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test evaluate!!(ne, (a=1.0, b=[2.0, 3.0])) == 6.0

        adtype = DummyADType()
        prepared = prepare(adtype, DummyProblem(), zeros(3))
        @test evaluate!!(prepared, [0.5, 1.5, 2.5]) ≈ 4.5
    end
end
