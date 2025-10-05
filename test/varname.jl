using Accessors
using InvertedIndices
using OffsetArrays
using LinearAlgebra: LowerTriangular, UpperTriangular, cholesky

using AbstractPPL: ⊑, ⊒, ⋢, ⋣, ≍

using AbstractPPL: Accessors
using AbstractPPL.Accessors: IndexLens, PropertyLens, ⨟

macro test_strict_subsumption(x, y)
    quote
        @test $((varname(x))) ⊑ $((varname(y)))
        @test $((varname(x))) ⋣ $((varname(y)))
    end
end

function test_equal(o1::VarName{sym1}, o2::VarName{sym2}) where {sym1,sym2}
    return sym1 === sym2 && test_equal(o1.optic, o2.optic)
end
function test_equal(o1::ComposedFunction, o2::ComposedFunction)
    return test_equal(o1.inner, o2.inner) && test_equal(o1.outer, o2.outer)
end
function test_equal(o1::Accessors.IndexLens, o2::Accessors.IndexLens)
    return test_equal(o1.indices, o2.indices)
end
function test_equal(o1, o2)
    return o1 == o2
end

@testset "varnames" begin
    @testset "string and symbol conversion" begin
        vn1 = @varname x[1][2]
        @test string(vn1) == "x[1][2]"
        @test Symbol(vn1) == Symbol("x[1][2]")
    end

    @testset "equality and hashing" begin
        vn1 = @varname x[1][2]
        vn2 = @varname x[1][2]
        @test vn2 == vn1
        @test hash(vn2) == hash(vn1)
    end

    @testset "inspace" begin
        space = (:x, :y, @varname(z[1]), @varname(M[1:10, :]))
        @test inspace(@varname(x), space)
        @test inspace(@varname(y), space)
        @test inspace(@varname(x[1]), space)
        @test inspace(@varname(z[1][1]), space)
        @test inspace(@varname(z[1][:]), space)
        @test inspace(@varname(z[1][2:3:10]), space)
        @test inspace(@varname(M[[2, 3], 1]), space)
        @test_throws ErrorException inspace(@varname(M[:, 1:4]), space)
        @test inspace(@varname(M[1, [2, 4, 6]]), space)
        @test !inspace(@varname(z[2]), space)
        @test !inspace(@varname(z), space)
    end

    @testset "optic normalisation" begin
        # Push the limits a bit with four optics, one of which is identity, and
        # we'll parenthesise them in every possible way. (Some of these are
        # going to be equal even before normalisation, but we should test that
        # `normalise` works regardless of how Base or Accessors.jl define
        # associativity.)
        op1 = ((@o _.c) ∘ (@o _.b)) ∘ identity ∘ (@o _.a)
        op2 = (@o _.c) ∘ ((@o _.b) ∘ identity) ∘ (@o _.a)
        op3 = (@o _.c) ∘ (@o _.b) ∘ (identity ∘ (@o _.a))
        op4 = ((@o _.c) ∘ (@o _.b) ∘ identity) ∘ (@o _.a)
        op5 = (@o _.c) ∘ ((@o _.b) ∘ identity ∘ (@o _.a))
        op6 = (@o _.c) ∘ (@o _.b) ∘ identity ∘ (@o _.a)
        for op in (op1, op2, op3, op4, op5, op6)
            @test AbstractPPL.normalise(op) == (@o _.c) ∘ (@o _.b) ∘ (@o _.a)
        end
        # Prefix and unprefix also provide further testing for normalisation.
    end

    @testset "construction & concretization" begin
        i = 1:10
        j = 2:2:5
        @test @varname(A[1].b[i]) == @varname(A[1].b[1:10])
        @test @varname(A[j]) == @varname(A[2:2:5])

        @test @varname(A[:, 1][1 + 1]) == @varname(A[:, 1][2])
        @test(@varname(A[:, 1][2]) == VarName{:A}(@o(_[:, 1]) ⨟ @o(_[2])))

        # concretization
        y = zeros(10, 10)
        x = (a=[1.0 2.0; 3.0 4.0; 5.0 6.0],)

        @test @varname(y[begin, i], true) == @varname(y[1, 1:10])
        @test test_equal(@varname(y[:], true), @varname(y[1:100]))
        @test test_equal(@varname(y[:, begin], true), @varname(y[1:10, 1]))
        @test getoptic(AbstractPPL.concretize(@varname(y[:]), y)).indices[1] ===
            AbstractPPL.ConcretizedSlice(to_indices(y, (:,))[1])
        @test test_equal(@varname(x.a[1:end, end][:], true), @varname(x.a[1:3, 2][1:3]))
    end

    @testset "compose and opcompose" begin
        @test IndexLens(1) ∘ @varname(x.a) == @varname(x.a[1])
        @test @varname(x.a) ⨟ IndexLens(1) == @varname(x.a[1])

        @test @varname(x) ⨟ identity == @varname(x)
        @test identity ∘ @varname(x) == @varname(x)
        @test @varname(x.a) ⨟ identity == @varname(x.a)
        @test identity ∘ @varname(x.a) == @varname(x.a)
        @test @varname(x[1].b) ⨟ identity == @varname(x[1].b)
        @test identity ∘ @varname(x[1].b) == @varname(x[1].b)
    end

    @testset "get & set" begin
        x = (a=[1.0 2.0; 3.0 4.0; 5.0 6.0], b=1.0)
        @test get(x, @varname(a[1, 2])) == 2.0
        @test get(x, @varname(b)) == 1.0
        @test set(x, @varname(a[1, 2]), 10) == (a=[1.0 10.0; 3.0 4.0; 5.0 6.0], b=1.0)
        @test set(x, @varname(b), 10) == (a=[1.0 2.0; 3.0 4.0; 5.0 6.0], b=10.0)
    end

    @testset "subsumption with standard indexing" begin
        # x ⊑ x
        @test @varname(x) ⊑ @varname(x)
        @test @varname(x[1]) ⊑ @varname(x[1])
        @test @varname(x.a) ⊑ @varname(x.a)

        # x ≍ y
        @test @varname(x) ≍ @varname(y)
        @test @varname(x.a) ≍ @varname(y.a)
        @test @varname(a.x) ≍ @varname(a.y)
        @test @varname(x[1]) ≍ @varname(y[1])

        # x ∘ ℓ ⊑ x
        @test_strict_subsumption x.a x
        @test_strict_subsumption x[1] x
        @test_strict_subsumption x[2:2:5] x
        @test_strict_subsumption x[10, 20] x

        # x ∘ ℓ₁ ⊑ x ∘ ℓ₂ ⇔ ℓ₁ ⊑ ℓ₂
        @test_strict_subsumption x.a.b x.a
        @test_strict_subsumption x[1].a x[1]
        @test_strict_subsumption x.a[1] x.a
        @test_strict_subsumption x[1:10][2] x[1:10]

        @test_strict_subsumption x[1] x[1:10]
        @test_strict_subsumption x[1:5] x[1:10]
        @test_strict_subsumption x[4:6] x[1:10]

        @test_strict_subsumption x[[2, 3, 5]] x[[7, 6, 5, 4, 3, 2, 1]]

        @test_strict_subsumption x[:a][1] x[:a]

        # boolean indexing works as long as it is concretized
        A = rand(10, 10)
        @test @varname(A[iseven.(1:10), 1], true) ⊑ @varname(A[1:10, 1])
        @test @varname(A[iseven.(1:10), 1], true) ⋣ @varname(A[1:10, 1])

        # we can reasonably allow colons on the right side ("universal set")
        @test @varname(x[1]) ⊑ @varname(x[:])
        @test @varname(x[1:10, 1]) ⊑ @varname(x[:, 1:10])
        @test_throws ErrorException (@varname(x[:]) ⊑ @varname(x[1]))
        @test_throws ErrorException (@varname(x[:]) ⊑ @varname(x[:]))
    end

    @testset "non-standard indexing" begin
        A = rand(10, 10)
        @test test_equal(
            @varname(A[1, Not(3)], true), @varname(A[1, [1, 2, 4, 5, 6, 7, 8, 9, 10]])
        )

        B = OffsetArray(A, -5, -5) # indices -4:5×-4:5
        @test test_equal(@varname(B[1, :], true), @varname(B[1, -4:5]))
    end
    @testset "type stability" begin
        @inferred VarName{:a}()
        @inferred VarName{:a}(IndexLens(1))
        @inferred VarName{:a}(IndexLens(1, 2))
        @inferred VarName{:a}(PropertyLens(:b))
        @inferred VarName{:a}(Accessors.opcompose(IndexLens(1), PropertyLens(:b)))

        b = (a=[1, 2, 3],)
        @inferred get(b, @varname(a[1]))
        @inferred Accessors.set(b, @varname(a[1]), 10)

        c = (b=(a=[1, 2, 3],),)
        @inferred get(c, @varname(b.a[1]))
        @inferred Accessors.set(c, @varname(b.a[1]), 10)
    end

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

    @testset "head, tail, init, last" begin
        @testset "specification" begin
            @test AbstractPPL._head(@o _.a.b.c) == @o _.a
            @test AbstractPPL._tail(@o _.a.b.c) == @o _.b.c
            @test AbstractPPL._init(@o _.a.b.c) == @o _.a.b
            @test AbstractPPL._last(@o _.a.b.c) == @o _.c

            @test AbstractPPL._head(@o _[1][2][3]) == @o _[1]
            @test AbstractPPL._tail(@o _[1][2][3]) == @o _[2][3]
            @test AbstractPPL._init(@o _[1][2][3]) == @o _[1][2]
            @test AbstractPPL._last(@o _[1][2][3]) == @o _[3]

            @test AbstractPPL._head(@o _.a) == @o _.a
            @test AbstractPPL._tail(@o _.a) == identity
            @test AbstractPPL._init(@o _.a) == identity
            @test AbstractPPL._last(@o _.a) == @o _.a

            @test AbstractPPL._head(@o _[1]) == @o _[1]
            @test AbstractPPL._tail(@o _[1]) == identity
            @test AbstractPPL._init(@o _[1]) == identity
            @test AbstractPPL._last(@o _[1]) == @o _[1]

            @test AbstractPPL._head(identity) == identity
            @test AbstractPPL._tail(identity) == identity
            @test AbstractPPL._init(identity) == identity
            @test AbstractPPL._last(identity) == identity
        end

        @testset "composition" begin
            varnames = (
                @varname(x),
                @varname(x[1]),
                @varname(x.a),
                @varname(x.a.b),
                @varname(x[1].a),
            )
            for vn in varnames
                optic = getoptic(vn)
                @test AbstractPPL.normalise(
                    AbstractPPL._last(optic) ∘ AbstractPPL._init(optic)
                ) == optic
                @test AbstractPPL.normalise(
                    AbstractPPL._tail(optic) ∘ AbstractPPL._head(optic)
                ) == optic
            end
        end
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
            x = (a=1.0, b=2.0)
            @test Set(varname_leaves(@varname(x), x)) == Set([@varname(x.a), @varname(x.b)])
            @test Set(collect(varname_and_value_leaves(@varname(x), x))) ==
                Set([(@varname(x.a), x.a), (@varname(x.b), x.b)])
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
    end
end
