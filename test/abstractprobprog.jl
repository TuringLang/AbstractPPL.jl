using AbstractPPL
using Random
using Test

mutable struct RandModel <: AbstractProbabilisticProgram
    rng
    T
end

function Base.rand(rng::Random.AbstractRNG, ::Type{T}, model::RandModel) where {T}
    model.rng = rng
    model.T = T
    return nothing
end

@testset "AbstractProbabilisticProgram" begin
    @testset "rand defaults" begin
        model = RandModel(nothing, nothing)
        rand(model)
        @test model.rng == Random.default_rng()
        @test model.T === NamedTuple
        rngs = [Random.default_rng(), Random.MersenneTwister(42)]
        Ts = [NamedTuple, Dict]
        @testset for T in Ts
            model = RandModel(nothing, nothing)
            rand(T, model)
            @test model.rng == Random.default_rng()
            @test model.T === T
        end
        @testset for rng in rngs
            model = RandModel(nothing, nothing)
            rand(rng, model)
            @test model.rng === rng
            @test model.T === NamedTuple
        end
    end
end
