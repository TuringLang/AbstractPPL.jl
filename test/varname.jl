using Accessors
using InvertedIndices
using OffsetArrays

using AbstractPPL: ⊑, ⊒, ⋢, ⋣, ≍

using AbstractPPL: Accessors
using AbstractPPL.Accessors: IndexLens, PropertyLens

macro test_strict_subsumption(x, y)
    quote
        @test $((varname(x))) ⊑ $((varname(y)))
        @test $((varname(x))) ⋣ $((varname(y)))
    end
end

@testset "varnames" begin
    @testset "construction & concretization" begin
        i = 1:10
        j = 2:2:5
        @test @varname(A[1].b[i]) == @varname(A[1].b[1:10])
        @test @varname(A[j]) == @varname(A[2:2:5])
        
        @test @varname(A[:, 1][1+1]) == @varname(A[:, 1][2])
        @test(@varname(A[:, 1][2]) ==
            VarName{:A}(@o(_[:, 1]) ⨟ @o(_[2])))

        # concretization
        y = zeros(10, 10)
        x = (a = [1.0 2.0; 3.0 4.0; 5.0 6.0], );

        @test @varname(y[begin, i], true) == @varname(y[1, 1:10])
        @test get(y, @varname(y[:], true)) ==  get(y, @varname(y[1:100]))
        @test get(y, @varname(y[:, begin], true)) == get(y, @varname(y[1:10, 1]))
        @test getoptic(AbstractPPL.concretize(@varname(y[:]), y)).indices[1] ===
            AbstractPPL.ConcretizedSlice(to_indices(y, (:,))[1])
        @test get(x, @varname(x.a[1:end, end][:], true)) == get(x, @varname(x.a[1:3,2][1:3]))
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
        
        @test_strict_subsumption x[[2,3,5]] x[[7,6,5,4,3,2,1]]

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
        @test get(A, @varname(A[1, Not(3)], true)) == get(A, @varname(A[1, [1, 2, 4, 5, 6, 7, 8, 9, 10]]))
        
        B = OffsetArray(A, -5, -5) # indices -4:5×-4:5
        @test collect(get(B, @varname(B[1, :], true))) == collect(get(B, @varname(B[1, -4:5])))

    end

    @testset "type stability" begin
        @inferred VarName{:a}()
        @inferred VarName{:a}(IndexLens(1))
        @inferred VarName{:a}(IndexLens(1, 2))
        @inferred VarName{:a}(PropertyLens(:b))
        @inferred VarName{:a}(Accessors.opcompose(IndexLens(1), PropertyLens(:b)))

        a = [1, 2, 3]
        @inferred get(a, @varname(a[1]))

        b = (a=[1, 2, 3],)
        @inferred get(b, @varname(b.a[1]))
        @inferred Accessors.set(b, @varname(a[1]), 10)

        c = (b=(a=[1, 2, 3],),)
        @inferred get(c, @varname(c.b.a[1]))
        @inferred Accessors.set(c, @varname(b.a[1]), 10)
    end
end
