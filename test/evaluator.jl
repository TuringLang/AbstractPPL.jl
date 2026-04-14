using AbstractPPL
using ADTypes: ADTypes
using Test

struct DummyProblem end

struct DummyPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(problem::DummyProblem, prototype::NamedTuple)
    return DummyPrepared(keys(prototype))
end

function (p::DummyPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys ||
        error("expected fields $(p.prototype_keys), got $(keys(values))")
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

struct DummyADPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(
    ::ADTypes.AbstractADType, problem::DummyProblem, prototype::NamedTuple
)
    return DummyADPrepared(keys(prototype))
end

function (p::DummyADPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys ||
        error("expected fields $(p.prototype_keys), got $(keys(values))")
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

AbstractPPL.capabilities(::Type{DummyADPrepared}) = DerivativeOrder{1}()

function AbstractPPL.value_and_gradient(p::DummyADPrepared, values::NamedTuple)
    v = p(values)
    grad = map(x -> x isa AbstractArray ? ones(size(x)) : 1.0, values)
    return (v, grad)
end

struct DummyVectorPrepared
    dim::Int
end

AbstractPPL.dimension(p::DummyVectorPrepared) = p.dim

function (p::DummyVectorPrepared)(x::AbstractVector)
    length(x) == p.dim || error("expected vector of length $(p.dim)")
    return sum(x)
end

@testset "Evaluator interface" begin
    @testset "DerivativeOrder" begin
        @test_throws ArgumentError DerivativeOrder{3}()
        @test_throws ArgumentError DerivativeOrder{-1}()
        @test DerivativeOrder{0}() < DerivativeOrder{1}()
        @test DerivativeOrder{1}() >= DerivativeOrder{1}()
        @test DerivativeOrder{1}() < DerivativeOrder{2}()
        @test !(DerivativeOrder{2}() < DerivativeOrder{1}())
    end

    @testset "capabilities default" begin
        # Any type without a capabilities method should return DerivativeOrder{0}
        @test capabilities(Int) == DerivativeOrder{0}()
        @test capabilities(42) == DerivativeOrder{0}()
        @test capabilities(DummyPrepared((:x,))) == DerivativeOrder{0}()
    end

    @testset "prepare (structural)" begin
        problem = DummyProblem()
        prototype = (x=0.0, y=[1.0, 2.0])
        prepared = prepare(problem, prototype)
        @test prepared isa DummyPrepared
        @test prepared.prototype_keys == (:x, :y)

        lp = prepared((x=0.5, y=[1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        @test_throws Exception prepared((a=1.0, b=2.0))
    end

    @testset "prepare (AD-aware)" begin
        problem = DummyProblem()
        prototype = (x=0.0, y=[1.0, 2.0])
        adtype = ADTypes.AutoForwardDiff()
        prepared = prepare(adtype, problem, prototype)
        @test prepared isa DummyADPrepared
        @test capabilities(prepared) == DerivativeOrder{1}()

        lp = prepared((x=0.5, y=[1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        val, grad = value_and_gradient(prepared, (x=0.5, y=[1.5, 2.5]))
        @test val ≈ 0.5 + 1.5 + 2.5
        @test grad.x ≈ 1.0
        @test grad.y ≈ [1.0, 1.0]
    end

    @testset "dimension and vector adapter" begin
        prepared = DummyVectorPrepared(3)
        @test dimension(prepared) == 3
        @test prepared(ones(3)) ≈ 3.0
        @test_throws Exception prepared(ones(5))
    end

    @testset "flatten / unflatten" begin
        nt = (x=1.0, y=[2.0, 3.0])
        v = AbstractPPL.flatten_to_vec(nt)
        @test v == [1.0, 2.0, 3.0]
        nt2 = AbstractPPL.unflatten_from_vec(nt, v)
        @test nt2.x == 1.0
        @test nt2.y == [2.0, 3.0]

        # Nested NamedTuple
        nt3 = (a=0.5, b=(c=1.0, d=[2.0, 3.0]))
        v3 = AbstractPPL.flatten_to_vec(nt3)
        @test v3 == [0.5, 1.0, 2.0, 3.0]
        nt3r = AbstractPPL.unflatten_from_vec(nt3, v3)
        @test nt3r.a == 0.5
        @test nt3r.b.c == 1.0
        @test nt3r.b.d == [2.0, 3.0]

        # Matrix
        nt4 = (x=[1.0 2.0; 3.0 4.0],)
        v4 = AbstractPPL.flatten_to_vec(nt4)
        @test length(v4) == 4
        nt4r = AbstractPPL.unflatten_from_vec(nt4, v4)
        @test nt4r.x == [1.0 2.0; 3.0 4.0]

        # P2: element type is preserved (not coerced to Float64)
        nt_f32 = (x=Float32(1.0), y=Float32[2.0, 3.0])
        v_f32 = AbstractPPL.flatten_to_vec(nt_f32)
        @test eltype(v_f32) == Float32
        nt_big = (x=big(1.0),)
        v_big = AbstractPPL.flatten_to_vec(nt_big)
        @test eltype(v_big) == BigFloat

        # P1: overlong vector is rejected
        @test_throws DimensionMismatch AbstractPPL.unflatten_from_vec(
            nt, [1.0, 2.0, 3.0, 99.0]
        )
    end
end
