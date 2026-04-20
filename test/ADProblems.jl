using AbstractPPL
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
    ::DummyADType, problem::DummyProblem, x::AbstractVector{<:AbstractFloat}
)
    return DummyADPrepared(length(x))
end

function (p::DummyADPrepared)(x::AbstractVector{<:AbstractFloat})
    length(x) == p.dim || error("expected vector of length $(p.dim)")
    return sum(x)
end

AbstractPPL.capabilities(::Type{DummyADPrepared}) = DerivativeOrder{1}()

function AbstractPPL.value_and_gradient(
    p::DummyADPrepared, x::AbstractVector{<:AbstractFloat}
)
    return (sum(x), ones(length(x)))
end

struct DummyVectorPrepared
    dim::Int
end

AbstractPPL.dimension(p::DummyVectorPrepared) = p.dim

function (p::DummyVectorPrepared)(x::AbstractVector)
    length(x) == p.dim || error("expected vector of length $(p.dim)")
    return sum(x)
end

@testset "ADProblem interface" begin
    @testset "explicit evaluator shapes" begin
        ve = AbstractPPL.ADProblems.VectorEvaluator(sum, 3)
        @test ve([1.0, 2.0, 3.0]) == 6.0
        @test dimension(ve) == 3
        @test_throws DimensionMismatch ve([1.0, 2.0])
        @test_throws MethodError ve([1, 2, 3])

        ne = AbstractPPL.ADProblems.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test ne((a=1.0, b=[2.0, 3.0])) == 6.0
        @test ne.inputspec == (a=0.0, b=zeros(2))
        err = try
            dimension(ne)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin(
            "only available for evaluators prepared with a vector", sprint(showerror, err)
        )
        @test_throws MethodError ne([1.0, 2.0, 3.0])
    end

    @testset "DerivativeOrder" begin
        err = try
            DerivativeOrder{3}()
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        @test occursin("must be 0, 1, or 2", sprint(showerror, err))
        @test_throws ArgumentError DerivativeOrder{-1}()
        @test DerivativeOrder{0}() < DerivativeOrder{1}()
        @test DerivativeOrder{1}() >= DerivativeOrder{1}()
        @test DerivativeOrder{1}() < DerivativeOrder{2}()
        @test !(DerivativeOrder{2}() < DerivativeOrder{1}())
    end

    @testset "capabilities default" begin
        @test capabilities(Int) == DerivativeOrder{0}()
        @test capabilities(42) == DerivativeOrder{0}()
        @test capabilities(DummyPrepared((:x,))) == DerivativeOrder{0}()
        @test capabilities(DummyPrepared((:x,))) < DerivativeOrder{1}()
    end

    @testset "prepare (structural)" begin
        problem = DummyProblem()
        values = (x=0.0, y=[1.0, 2.0])
        prepared = prepare(problem, values)
        @test prepared isa DummyPrepared
        @test prepared.prototype_keys == (:x, :y)

        lp = prepared((x=0.5, y=[1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        @test_throws Exception prepared((a=1.0, b=2.0))
    end

    @testset "prepare (AD-aware)" begin
        problem = DummyProblem()
        x0 = zeros(3)
        adtype = DummyADType()
        prepared = prepare(adtype, problem, x0)
        @test prepared isa DummyADPrepared
        @test capabilities(prepared) == DerivativeOrder{1}()

        x = [0.5, 1.5, 2.5]
        @test prepared(x) ≈ 0.5 + 1.5 + 2.5

        val, grad = value_and_gradient(prepared, x)
        @test val ≈ 0.5 + 1.5 + 2.5
        @test grad ≈ [1.0, 1.0, 1.0]
    end

    @testset "missing AD package extensions" begin
        problem = DummyProblem()
        x0 = zeros(3)

        for adtype in (
            ADTypes.AutoForwardDiff(),
            ADTypes.AutoEnzyme(),
            ADTypes.AutoMooncake(),
            ADTypes.AutoMooncakeForward(),
        )
            err = try
                AbstractPPL.ADProblems.prepare(adtype, problem, x0)
                nothing
            catch err
                err
            end
            @test err isa ArgumentError
            @test occursin(
                "requires loading the corresponding AD backend", sprint(showerror, err)
            )
        end
    end

    @testset "dimension and vector adapter" begin
        prepared = DummyVectorPrepared(3)
        @test dimension(prepared) == 3
        @test prepared(ones(3)) ≈ 3.0
        @test_throws Exception prepared(ones(5))
    end

    @testset "flatten / unflatten edge cases" begin
        empty = NamedTuple()
        @test AbstractPPL.Utils.flatten_to!!(nothing, empty) == Float64[]
        @test AbstractPPL.Utils.unflatten_to!!(empty, Float64[]) == empty

        view_values = (x=@view([1.0, 2.0, 3.0][2:3]),)
        flat = AbstractPPL.Utils.flatten_to!!(nothing, view_values)
        rebuilt = AbstractPPL.Utils.unflatten_to!!(view_values, flat)
        @test collect(rebuilt.x) == [2.0, 3.0]
        @test axes(rebuilt.x) == axes(view_values.x)
        @test parent(rebuilt.x) == [2.0, 3.0]
    end
end
