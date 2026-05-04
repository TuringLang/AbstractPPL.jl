using AbstractPPL
using AbstractPPL.Evaluators: flatten_to!!, unflatten_to!!
using LinearAlgebra: Symmetric
using OffsetArrays: OffsetArray
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

    @testset "heterogeneous container preserves leaf types" begin
        # The flat buffer widens to ComplexF64, but `unflatten_to!!` rebuilds
        # leaves using `x`'s types, so the round-trip preserves `typeof(x)`.
        x = (1.0, [2.0, 3.0], (4.0 + 1.0im,))
        x2 = unflatten_to!!(x, flatten_to!!(nothing, x))
        @test x2 == x
        @test typeof(x2) == typeof(x)
    end

    @testset "check_eltype opt-in warning" begin
        x = (a=1.0, b=[2.0, 3.0])
        # buf eltype is ComplexF64, but `flat_eltype(x) == Float64` — a
        # mismatch that should warn only with `check_eltype=true`. Imag parts
        # are zero, so the convert succeeds.
        buf = ComplexF64[1.0, 2.0, 3.0]
        x2 = @test_logs unflatten_to!!(x, buf)  # default: silent
        @test x2 == x
        @test typeof(x2) == typeof(x)
        x3 = @test_logs (:warn, r"differs from") unflatten_to!!(x, buf; check_eltype=true)
        @test x3 == x
        @test typeof(x3) == typeof(x)
    end

    @testset "non-one-based buf rejected" begin
        # Non-one-based `buf` fails `require_one_based_indexing` even when `x`
        # is a plain `Array`. (Non-one-based `x` is covered by the catch-all
        # in the next testset.)
        @test_throws ArgumentError flatten_to!!(OffsetArray(zeros(3), 0:2), [1.0, 2.0, 3.0])
        @test_throws ArgumentError unflatten_to!!(
            [1.0, 2.0, 3.0], OffsetArray(zeros(3), 0:2)
        )
    end

    @testset "non-Array AbstractArrays rejected" begin
        for x in (
            Symmetric([1.0 2.0; 2.0 3.0]),
            @view([1.0, 2.0, 3.0, 4.0][1:4]),
            OffsetArray([1.0, 2.0, 3.0], 0:2),
        )
            @test_throws r"cannot be flattened" flatten_to!!(nothing, x)
            @test_throws r"cannot be flattened" unflatten_to!!(x, [1.0, 2.0, 3.0, 4.0])
        end
    end

    @testset "buffer length mismatch" begin
        @test_throws r"Expected a vector of length 4" flatten_to!!(
            Vector{Float64}(undef, 3), zeros(2, 2)
        )
    end

    @testset "vector length mismatch" begin
        x = (a=1.0, b=[2.0, 3.0])
        @test_throws r"Expected a vector of length 3" unflatten_to!!(x, [1.0, 2.0])
    end

    @testset "edge cases" begin
        empty = NamedTuple()
        @test flatten_to!!(nothing, empty) == Float64[]
        @test unflatten_to!!(empty, Float64[]) == empty
    end

    @testset "unflatten_to!! type stability" begin
        @inferred unflatten_to!!((a=1.0, b=2.0, c=3.0), zeros(3))
        @inferred unflatten_to!!((a=1.0, b=[2.0, 3.0], c=3.0), zeros(4))
        @inferred unflatten_to!!((a=(p=1.0, q=2.0), b=3.0), zeros(3))
        @inferred unflatten_to!!((a=1.0, b=(2.0, 3.0)), zeros(3))
        @inferred unflatten_to!!(NamedTuple(), Float64[])
        @inferred unflatten_to!!((1.0, [2.0, 3.0], 3.0), zeros(4))
    end
end
