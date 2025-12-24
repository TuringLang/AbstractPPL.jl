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

    @testset "decomposition" begin
        @testset "specification" begin
            @test ohead(@opticof _.a.b.c) == @opticof _.a
            @test otail(@opticof _.a.b.c) == @opticof _.b.c
            @test oinit(@opticof _.a.b.c) == @opticof _.a.b
            @test olast(@opticof _.a.b.c) == @opticof _.c

            @test ohead(@opticof _[1][2][3]) == @opticof _[1]
            @test otail(@opticof _[1][2][3]) == @opticof _[2][3]
            @test oinit(@opticof _[1][2][3]) == @opticof _[1][2]
            @test olast(@opticof _[1][2][3]) == @opticof _[3]

            @test ohead(@opticof _.a) == @opticof _.a
            @test otail(@opticof _.a) == @opticof _
            @test oinit(@opticof _.a) == @opticof _
            @test olast(@opticof _.a) == @opticof _.a

            @test ohead(@opticof _[1]) == @opticof _[1]
            @test otail(@opticof _[1]) == @opticof _
            @test oinit(@opticof _[1]) == @opticof _
            @test olast(@opticof _[1]) == @opticof _[1]

            @test ohead(@opticof _) == @opticof _
            @test otail(@opticof _) == @opticof _
            @test oinit(@opticof _) == @opticof _
            @test olast(@opticof _) == @opticof _
        end

        @testset "invariants" begin
            optics = (
                @opticof(_),
                @opticof(_[1]),
                @opticof(_.a),
                @opticof(_.a.b),
                @opticof(_[1].a),
                @opticof(_[1, x=1].a),
                @opticof(_[].a[:]),
            )
            for optic in optics
                @test olast(optic) ∘ oinit(optic) == optic
                @test otail(optic) ∘ ohead(optic) == optic
            end
        end
    end

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

        @testset "no indices" begin
            x = [0.0]
            @test @opticof(_[])(x) == x[]
            @test set(x, @opticof(_[]), 9.0) == [9.0]
            x = 0.0
            @test @opticof(_[])(x) == x[]
            @test set(x, @opticof(_[]), 9.0) == 9.0
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

            # DimArray selectors
            dimarray = DD.DimArray(randn(2, 3), (DD.X, DD.Y))
            @test @opticof(_[DD.X(1)])(dimarray) == dimarray[DD.X(1)]

            # Symbols on NamedTuples
            nt = (a=10, b=20, c=30)
            @test @opticof(_[:a])(nt) == nt[:a]
            @test set(nt, @opticof(_[:b]), 99) == (a=10, b=99, c=30)

            # Strings on Dicts
            dict = Dict("one" => 1, "two" => 2)
            @test @opticof(_["two"])(dict) == dict["two"]
            @test set(dict, @opticof(_["two"]), 22) == Dict("one" => 1, "two" => 22)
        end

        @testset "keyword arguments to getindex" begin
            dimarray = DD.DimArray([0.0 1.0; 2.0 3.0], (:x, :y))
            @test @opticof(_[x=1])(dimarray) == dimarray[x=1]
            @test set(dimarray, @opticof(_[y=2]), [9.0; 8.0]) ==
                DD.DimArray([0.0 9.0; 2.0 8.0], (:x, :y))
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

    @testset "mutating versions" begin
        @testset "arrays" begin
            @testset "static index" begin
                x = zeros(4)
                old_objid = objectid(x)
                optic = with_mutation(@opticof(_[2]))
                @test optic(x) === x[2]
                set(x, optic, 1.0)
                @test x[2] == 1.0
                @test x == [0.0, 1.0, 0.0, 0.0]
                @test objectid(x) == old_objid
            end

            @testset "dynamic index" begin
                x = zeros(2, 2)
                old_objid = objectid(x)
                optic = with_mutation(@opticof(_[begin, end]))
                @test optic(x) === x[begin, end]
                set(x, optic, 2.0)
                @test x[begin, end] == 2.0
                @test x == [0.0 2.0; 0.0 0.0]
                @test objectid(x) == old_objid
            end

            @testset "DimArray, setting a single element" begin
                dimarray = DD.DimArray(zeros(2, 3), (DD.X, DD.Y))
                old_objid = objectid(dimarray)
                optic = with_mutation(@opticof(_[DD.X(1), DD.Y(1)]))
                @test optic(dimarray) == dimarray[DD.X(1), DD.Y(1)]
                dimarray = set(dimarray, optic, 1.0)
                @test dimarray[DD.X(1), DD.Y(1)] == 1.0
                @test collect(dimarray) == [1.0 0.0 0.0; 0.0 0.0 0.0]
                @test objectid(dimarray) == old_objid
            end

            @testset "keyword index" begin
                x = DD.DimArray(zeros(2, 2), (:x, :y))
                old_objid = objectid(x)
                optic = with_mutation(@opticof(_[x=1, y=2]))
                @test optic(x) === x[x=1, y=2]
                set(x, optic, 2.0)
                @test x[x=1, y=2] == 2.0
                @test collect(x) == [0.0 2.0; 0.0 0.0]
                @test objectid(x) == old_objid
            end
        end

        @testset "dicts" begin
            x = Dict("a" => 1, "b" => 2)
            old_objid = objectid(x)
            optic = with_mutation(@opticof(_["b"]))
            @test optic(x) === x["b"]
            set(x, optic, 99)
            @test x["b"] == 99
            @test x == Dict("a" => 1, "b" => 99)
            @test objectid(x) == old_objid
        end

        @testset "mutable structs" begin
            mutable struct MutableStruct
                a::Int
                b::Float64
            end
            x = MutableStruct(3, 1.5)
            old_objid = objectid(x)
            optic = with_mutation(@opticof(_.a))
            @test optic(x) === x.a
            set(x, optic, 10)
            @test x.a == 10
            @test x.b == 1.5
            @test objectid(x) == old_objid
        end

        @testset "fallback for immutable data" begin
            @testset "NamedTuple" begin
                s = (a=1, b=2)
                old_objid = objectid(s)
                optic = with_mutation(@opticof(_.a))
                @test optic(s) === s.a
                s2 = set(s, optic, 10)
                @test s2 == (a=10, b=2)
                @test s == (a=1, b=2)
                @test objectid(s) == old_objid
            end

            # NOTE(penelopeysm): This SHOULD really mutate. It is not an error with
            # AbstractPPL, though, it is an interface problem between BangBang and
            # DimensionalData (essentially BangBang can't detect that DimArray is mutable).
            # 
            # To be precise, the test fails because BangBang thinks that `DD.Y(1)` is an
            # index that extracts a single element from the DimArray. For example, this
            # would be the case if it was dimarray[1]. So, BangBang thinks that you can't
            # set an array there, and so it falls back to the immutable behavior.
            # Specifically, it's this line that returns false:
            # https://github.com/JuliaFolds2/BangBang.jl/blob/e92b4c1673a686533b5f9724a198b63d8974d52f/src/base.jl#L528
            @testset "DimArray, setting a vector" begin
                dimarray = DD.DimArray(zeros(2, 3), (DD.X, DD.Y))
                old_objid = objectid(dimarray)
                optic = with_mutation(@opticof(_[DD.Y(1)]))
                @test optic(dimarray) == dimarray[DD.Y(1)]
                dimarray = set(dimarray, optic, [1.0, 2.0])
                @test collect(dimarray[DD.Y(1)]) == [1.0; 2.0]
                @test collect(dimarray) == [1.0 0.0 0.0; 2.0 0.0 0.0]
                # @test objectid(dimarray) == old_objid
            end

            @testset "tuple" begin
                s = (3, 1.5)
                old_objid = objectid(s)
                optic = with_mutation(@opticof(_[1]))
                @test optic(s) === s[1]
                s2 = set(s, optic, 10)
                @test s2 == (10, 1.5)
                @test s == (3, 1.5)
                @test objectid(s) == old_objid
            end

            @testset "struct" begin
                struct SampleStructAgain
                    a::Int
                    b::Float64
                end
                s = SampleStructAgain(3, 1.5)
                old_objid = objectid(s)
                optic = with_mutation(@opticof(_.a))
                @test optic(s) === s.a
                s2 = set(s, optic, 10)
                @test s2 == SampleStructAgain(10, 1.5)
                @test s == SampleStructAgain(3, 1.5)
                @test objectid(s) == old_objid
            end
        end
    end
end

end # module
