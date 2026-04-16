using AbstractPPL
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using Test

@testset "vectorisation utilities" begin
    @testset "scalar round-trip" begin
        x = 1.5
        v = flatten_to!!(nothing, x)
        @test v == [1.5]
        @test unflatten_to!!(x, v) == 1.5

        z = 1.0 + 2.0im
        vz = flatten_to!!(nothing, z)
        @test vz == ComplexF64[1.0 + 2.0im]
        @test unflatten_to!!(z, vz) == z
    end

    @testset "array round-trip" begin
        x = [1.0 2.0; 3.0 4.0]
        v = flatten_to!!(nothing, x)
        @test v == [1.0, 3.0, 2.0, 4.0]
        @test unflatten_to!!(x, v) == x

        z = ComplexF64[1.0 + 1.0im, 2.0 + 0.0im]
        vz = flatten_to!!(nothing, z)
        @test vz == z
        @test unflatten_to!!(z, vz) == z
    end

    @testset "tuple round-trip" begin
        x = (1.0, [2.0, 3.0], (4.0 + 1.0im,))
        v = flatten_to!!(nothing, x)
        @test v == ComplexF64[1.0, 2.0, 3.0, 4.0 + 1.0im]
        @test unflatten_to!!(x, v) == x
    end

    @testset "named tuple round-trip" begin
        x = (a=1.0, b=([2.0, 3.0], (c=4.0 + 1.0im,)))
        v = flatten_to!!(nothing, x)
        @test v == ComplexF64[1.0, 2.0, 3.0, 4.0 + 1.0im]
        @test unflatten_to!!(x, v) == x
    end

    @testset "buffer length mismatch" begin
        err = try
            flatten_to!!(Vector{Float64}(undef, 3), zeros(2, 2))
            nothing
        catch err
            err
        end
        @test err isa DimensionMismatch
        @test occursin("Expected a vector of length 4", sprint(showerror, err))
    end

    @testset "vector length mismatch" begin
        x = (a=1.0, b=[2.0, 3.0])
        err = try
            unflatten_to!!(x, [1.0, 2.0])
            nothing
        catch err
            err
        end
        @test err isa DimensionMismatch
        @test occursin("Expected a vector of length 3", sprint(showerror, err))
    end
end
