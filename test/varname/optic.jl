module OpticTests

using Test
using DimensionalData: DimensionalData as DD
using AbstractPPL

@testset "varname/optic.jl" verbose = true begin
    # Note that much of the functionality in optic.jl is tested by varname.jl (for example,
    # pretty-printing VarNames essentially boils down to pretty-printing optics). So, this
    # file focuses on tests that are specific to optics.
    @testset "composition" begin
        @testset "with identity" begin
            i = AbstractPPL.Iden()
            o = @opticof(_.a.b)
            @test i ∘ i == i
            @test i ∘ o == o
            @test o ∘ i == o
        end

        o1 = @opticof(_.a.b)
        o2 = @opticof(_[1][2])
        @test o1 ∘ o2 == @opticof(_[1][2].a.b)
        @test o2 ∘ o1 == @opticof(_.a.b[1][2])
        @test cat(o1, o2) == @opticof(_.a.b[1][2])
        @test cat(o2, o1) == @opticof(_[1][2].a.b)
        @test cat(o1, o2, o2, o1) == @opticof(_.a.b[1][2][1][2].a.b)
    end

    # TODO
    @testset "decomposition" begin end

    @testset "getting and setting" begin
        @testset "basic" begin
            v = (a=(b=42, c=3.14), d=[0.0 1.0; 2.0 3.0])
            @test @opticof(_.a)(v) == v.a
            @test set(v, @opticof(_.a), nothing) == (a=nothing, d=v.d)
            @test @opticof(_.a.b)(v) == v.a.b
            @test set(v, @opticof(_.a.b), 100) == (a=(b=100, c=v.a.c), d=v.d)
            @test @opticof(_.a.c)(v) == v.a.c
            @test set(v, @opticof(_.a.c), 2.71) == (a=(b=v.a.b, c=2.71), d=v.d)
            @test @opticof(_.d)(v) == v.d
            @test set(v, @opticof(_.d), zeros(2, 2)) == (a=v.a, d=zeros(2, 2))
            @test @opticof(_.d[1])(v) == v.d[1]
            @test set(v, @opticof(_.d[1]), 9.0) == (a=v.a, d=[9.0 1.0; 2.0 3.0])
            @test @opticof(_.d[2])(v) == v.d[2]
            @test set(v, @opticof(_.d[2]), 9.0) == (a=v.a, d=[0.0 1.0; 9.0 3.0])
            @test @opticof(_.d[3])(v) == v.d[3]
            @test set(v, @opticof(_.d[3]), 9.0) == (a=v.a, d=[0.0 9.0; 2.0 3.0])
            @test @opticof(_.d[4])(v) == v.d[4]
            @test set(v, @opticof(_.d[4]), 9.0) == (a=v.a, d=[0.0 1.0; 2.0 9.0])
            @test @opticof(_.d[:])(v) == v.d[:]
            @test set(v, @opticof(_.d[:]), fill(9.9, 2, 2)) == (a=v.a, d=fill(9.9, 2, 2))
        end

        @testset "dynamic indices" begin
            x = [0.0 1.0; 2.0 3.0]
            @test @opticof(_[begin])(x) == x[begin]
            @test set(x, @opticof(_[begin]), 9.0) == [9.0 1.0; 2.0 3.0]
            @test @opticof(_[end])(x) == x[end]
            @test set(x, @opticof(_[end]), 9.0) == [0.0 1.0; 2.0 9.0]
            @test @opticof(_[1:end, 2])(x) == x[1:end, 2]
            @test set(x, @opticof(_[1:end, 2]), [9.0; 8.0]) == [0.0 9.0; 2.0 8.0]
        end

        @testset "unusual indices" begin
            x = randn(3, 3)
            @test @opticof(_[1:2:4])(x) == x[1:2:4]
            @test @opticof(_[CartesianIndex(1, 1)])(x) == x[CartesianIndex(1, 1)]
            # `Not` is actually from InvertedIndices.jl (but re-exported by DimensionalData)
            @test @opticof(_[DD.Not(3)])(x) == x[DD.Not(3)]
            dimarray = DD.DimArray(randn(2, 3), (DD.X, DD.Y))
            @test @opticof(_[DD.X(1)])(dimarray) == dimarray[DD.X(1)]
            # TODO(penelopeysm): This doesn't support keyword arguments to getindex yet.
            # For example:
            # dimarray = DD.DimArray(randn(2, 3), (:x, :y))
            # @test @opticof(_[x=1])(dimarray) == dimarray[x=1]
        end

        struct SampleStruct
            a::Int
            b::Float64
        end
        s = SampleStruct(3, 1.5)
        @test @opticof(_.a)(s) == 3
        @test @opticof(_.b)(s) == 1.5
        @test set(s, @opticof(_.a), 10) == SampleStruct(10, s.b)
        @test set(s, @opticof(_.b), 2.5) == SampleStruct(s.a, 2.5)
    end
end

end # module
