using Accessors
using InvertedIndices
using OffsetArrays
using LinearAlgebra: LowerTriangular, UpperTriangular, cholesky

@testset "varnames" begin
    @testset "de/serialisation of VarNames" begin
        y = ones(10)
        z = ones(5, 2)
        vns = [
            @varname(x),
            @varname(ä),
            @varname(x.a),
            @varname(x.a.b),
            @varname(var"x.a"),
            @varname(x[1]),
            @varname(var"x[1]"),
            @varname(x[1:10]),
            @varname(x[1:3:10]),
            @varname(x[1, 2]),
            @varname(x[1, 2:5]),
            @varname(x[:]),
            @varname(x.a[1]),
            @varname(x.a[1:10]),
            @varname(x[1].a),
            @varname(y[:]),
            @varname(y[begin:end]),
            @varname(y[end]),
            @varname(y[:], false),
            @varname(y[:], true),
            @varname(z[:], false),
            @varname(z[:], true),
            @varname(z[:][:], false),
            @varname(z[:][:], true),
            @varname(z[:, :], false),
            @varname(z[:, :], true),
            @varname(z[2:5, :], false),
            @varname(z[2:5, :], true),
        ]
        for vn in vns
            @test string_to_varname(varname_to_string(vn)) == vn
        end

        # For this VarName, the {de,}serialisation works correctly but we must
        # test in a different way because equality comparison of structs with
        # vector fields (such as Accessors.IndexLens) compares the memory
        # addresses rather than the contents (thus vn_vec == vn_vec2 returns
        # false).
        vn_vec = @varname(x[[1, 2, 5, 6]])
        vn_vec2 = string_to_varname(varname_to_string(vn_vec))
        @test hash(vn_vec) == hash(vn_vec2)
    end

    @testset "de/serialisation of VarNames with custom index types" begin
        using OffsetArrays: OffsetArrays, Origin
        weird = Origin(4)(ones(10))
        vn = @varname(weird[:], true)

        # This won't work as we don't yet know how to handle OffsetArray
        @test_throws MethodError varname_to_string(vn)

        # Now define the relevant methods
        AbstractPPL.index_to_dict(o::OffsetArrays.IdOffsetRange{I,R}) where {I,R} = Dict(
            "type" => "OffsetArrays.OffsetArray",
            "parent" => AbstractPPL.index_to_dict(o.parent),
            "offset" => o.offset,
        )
        AbstractPPL.dict_to_index(::Val{Symbol("OffsetArrays.OffsetArray")}, d) =
            OffsetArrays.IdOffsetRange(AbstractPPL.dict_to_index(d["parent"]), d["offset"])

        # Serialisation should now work
        @test string_to_varname(varname_to_string(vn)) == vn
    end

    @testset "prefix and unprefix" begin
        @testset "basic cases" begin
            @test prefix(@varname(y), @varname(x)) == @varname(x.y)
            @test prefix(@varname(y), @varname(x[1])) == @varname(x[1].y)
            @test prefix(@varname(y), @varname(x.a)) == @varname(x.a.y)
            @test prefix(@varname(y[1]), @varname(x)) == @varname(x.y[1])
            @test prefix(@varname(y.a), @varname(x)) == @varname(x.y.a)

            @test unprefix(@varname(x.y[1]), @varname(x)) == @varname(y[1])
            @test unprefix(@varname(x[1].y), @varname(x[1])) == @varname(y)
            @test unprefix(@varname(x.a.y), @varname(x.a)) == @varname(y)
            @test unprefix(@varname(x.y.a), @varname(x)) == @varname(y.a)
            @test_throws ArgumentError unprefix(@varname(x.y.a), @varname(n))
            @test_throws ArgumentError unprefix(@varname(x.y.a), @varname(x[1]))
        end

        @testset "round-trip" begin
            # These seem similar to the ones above, but in the past they used
            # to error because of issues with un-normalised ComposedFunction
            # optics. We explicitly test round-trip (un)prefixing here to make
            # sure that there aren't any regressions.
            # This tuple is probably overkill, but the tests are super fast
            # anyway.
            vns = (
                @varname(p),
                @varname(q),
                @varname(r[1]),
                @varname(s.a),
                @varname(t[1].a),
                @varname(u[1].a.b),
                @varname(v.a[1][2].b.c.d[3])
            )
            for vn1 in vns
                for vn2 in vns
                    prefixed = prefix(vn1, vn2)
                    @test subsumes(vn2, prefixed)
                    unprefixed = unprefix(prefixed, vn2)
                    @test unprefixed == vn1
                end
            end
        end
    end

    @testset "varname{_and_value}_leaves" begin
        @testset "single value: float, int" begin
            x = 1.0
            @test Set(varname_leaves(@varname(x), x)) == Set([@varname(x)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x), x)])
            x = 2
            @test Set(varname_leaves(@varname(x), x)) == Set([@varname(x)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x), x)])
        end

        @testset "Vector" begin
            x = randn(2)
            @test Set(varname_leaves(@varname(x), x)) ==
                Set([@varname(x[1]), @varname(x[2])])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x[1]), x[1]), (@varname(x[2]), x[2])])
            x = [(; a=1), (; b=2)]
            @test Set(varname_leaves(@varname(x), x)) ==
                Set([@varname(x[1].a), @varname(x[2].b)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x[1].a), x[1].a), (@varname(x[2].b), x[2].b)])
        end

        @testset "Matrix" begin
            x = randn(2, 2)
            @test Set(varname_leaves(@varname(x), x)) == Set([
                @varname(x[1, 1]), @varname(x[1, 2]), @varname(x[2, 1]), @varname(x[2, 2])
            ])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) == Set([
                (@varname(x[1, 1]), x[1, 1]),
                (@varname(x[1, 2]), x[1, 2]),
                (@varname(x[2, 1]), x[2, 1]),
                (@varname(x[2, 2]), x[2, 2]),
            ])
        end

        @testset "Lower/UpperTriangular" begin
            x = randn(2, 2)
            xl = LowerTriangular(x)
            @test Set(varname_leaves(@varname(x), xl)) ==
                Set([@varname(x[1, 1]), @varname(x[2, 1]), @varname(x[2, 2])])
            @test Set(collect(varname_and_value_leaves(@varname(x), xl))) == Set([
                (@varname(x[1, 1]), x[1, 1]),
                (@varname(x[2, 1]), x[2, 1]),
                (@varname(x[2, 2]), x[2, 2]),
            ])
            xu = UpperTriangular(x)
            @test Set(varname_leaves(@varname(x), xu)) ==
                Set([@varname(x[1, 1]), @varname(x[1, 2]), @varname(x[2, 2])])
            @test Set(collect(varname_and_value_leaves(@varname(x), xu))) == Set([
                (@varname(x[1, 1]), x[1, 1]),
                (@varname(x[1, 2]), x[1, 2]),
                (@varname(x[2, 2]), x[2, 2]),
            ])
        end

        @testset "NamedTuple" begin
            x = (a=1.0, b=[2.0, 3.0])
            @test Set(varname_leaves(@varname(x), x)) ==
                Set([@varname(x.a), @varname(x.b[1]), @varname(x.b[2])])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) == Set([
                (@varname(x.a), x.a), (@varname(x.b[1]), x.b[1]), (@varname(x.b[2]), x.b[2])
            ])
        end

        @testset "Cholesky" begin
            x = cholesky([1.0 0.5; 0.5 1.0])
            @test Set(varname_leaves(@varname(x), x)) ==
                Set([@varname(x.U[1, 1]), @varname(x.U[1, 2]), @varname(x.U[2, 2])])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) == Set([
                (@varname(x.U[1, 1]), x.U[1, 1]),
                (@varname(x.U[1, 2]), x.U[1, 2]),
                (@varname(x.U[2, 2]), x.U[2, 2]),
            ])
        end

        @testset "fallback on other types, e.g. string" begin
            x = "a string"
            @test Set(varname_leaves(@varname(x), x)) == Set([@varname(x)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x), x)])
            x = 2
            @test Set(varname_leaves(@varname(x), x)) == Set([@varname(x)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x), x)])
        end
    end
end
