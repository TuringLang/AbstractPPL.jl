using AbstractPPL
using ADTypes: ADTypes
using Test

# A minimal concrete problem and prepared evaluator for testing the interface.
struct DummyProblem end

struct DummyPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(problem::DummyProblem, prototype::NamedTuple)
    return DummyPrepared(keys(prototype))
end

function (p::DummyPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys || error(
        "expected fields $(p.prototype_keys), got $(keys(values))"
    )
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

# A prepared evaluator that also supports gradients.
struct DummyADPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(
    ::ADTypes.AbstractADType, problem::DummyProblem, prototype::NamedTuple
)
    return DummyADPrepared(keys(prototype))
end

function (p::DummyADPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys || error(
        "expected fields $(p.prototype_keys), got $(keys(values))"
    )
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

AbstractPPL.capabilities(::Type{DummyADPrepared}) = DerivativeOrder{1}()

function AbstractPPL.value_and_gradient(p::DummyADPrepared, values::NamedTuple)
    v = p(values)
    grad = map(x -> x isa AbstractArray ? ones(size(x)) : 1.0, values)
    return (v, grad)
end

# A prepared evaluator with a vector adapter.
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
        @test DerivativeOrder{0}() isa DerivativeOrder{0}
        @test DerivativeOrder{1}() isa DerivativeOrder{1}
        @test DerivativeOrder{2}() isa DerivativeOrder{2}
        @test_throws ArgumentError DerivativeOrder{3}()
        @test_throws ArgumentError DerivativeOrder{-1}()
    end

    @testset "capabilities default" begin
        # Any type without a capabilities method should return DerivativeOrder{0}
        @test capabilities(Int) == DerivativeOrder{0}()
        @test capabilities(42) == DerivativeOrder{0}()
        @test capabilities(DummyPrepared((:x,))) == DerivativeOrder{0}()
    end

    @testset "prepare (structural)" begin
        problem = DummyProblem()
        prototype = (x = 0.0, y = [1.0, 2.0])
        prepared = prepare(problem, prototype)
        @test prepared isa DummyPrepared
        @test prepared.prototype_keys == (:x, :y)

        # Callable on matching structure
        lp = prepared((x = 0.5, y = [1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        # Structural mismatch throws
        @test_throws Exception prepared((a = 1.0, b = 2.0))
    end

    @testset "prepare (AD-aware)" begin
        problem = DummyProblem()
        prototype = (x = 0.0, y = [1.0, 2.0])
        adtype = ADTypes.AutoForwardDiff()
        prepared = prepare(adtype, problem, prototype)
        @test prepared isa DummyADPrepared
        @test capabilities(prepared) == DerivativeOrder{1}()

        # Callable
        lp = prepared((x = 0.5, y = [1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        # value_and_gradient
        val, grad = value_and_gradient(prepared, (x = 0.5, y = [1.5, 2.5]))
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

end
