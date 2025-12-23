module VarNameSubsumesTests

using AbstractPPL
using Test

@testset "varname/subsumes.jl" verbose = true begin
    @test subsumes(@varname(x), @varname(x))
    @test subsumes(@varname(x[1]), @varname(x[1]))
    @test subsumes(@varname(x.a), @varname(x.a))

    uncomparable(vn1, vn2) = !subsumes(vn1, vn2) && !subsumes(vn2, vn1)
    @test uncomparable(@varname(x), @varname(y))
    @test uncomparable(@varname(x.a), @varname(y.a))
    @test uncomparable(@varname(a.x), @varname(a.y))
    @test uncomparable(@varname(a.x[1]), @varname(a.x.z))
    @test uncomparable(@varname(x[1]), @varname(y[1]))
    @test uncomparable(@varname(x[1]), @varname(x.y))

    strictly_subsumes(vn1, vn2) = subsumes(vn1, vn2) && !subsumes(vn2, vn1)
    # Subsumption via field/indexing
    @test strictly_subsumes(@varname(x), @varname(x.a))
    @test strictly_subsumes(@varname(x), @varname(x[1]))
    @test strictly_subsumes(@varname(x), @varname(x[2:2:5]))
    @test strictly_subsumes(@varname(x), @varname(x[10, 20]))
    @test strictly_subsumes(@varname(x.a), @varname(x.a.b))
    @test strictly_subsumes(@varname(x[1]), @varname(x[1].a))
    @test strictly_subsumes(@varname(x.a), @varname(x.a[1]))
    @test strictly_subsumes(@varname(x[1:10]), @varname(x[1:10][2]))
    # Range subsumption
    @test strictly_subsumes(@varname(x[1:10]), @varname(x[1]))
    @test strictly_subsumes(@varname(x[1:10]), @varname(x[1:5]))
    @test strictly_subsumes(@varname(x[1:10]), @varname(x[4:6]))
    @test strictly_subsumes(@varname(x[1:10, 1:10]), @varname(x[1:5, 1:5]))
    @test strictly_subsumes(@varname(x[[7, 6, 5, 4, 3, 2, 1]]), @varname(x[[2, 3, 5]]))

    # TODO reenable
    # @test_strict_subsumption x[:a][1] x[:a]
    # # boolean indexing works as long as it is concretized
    # A = rand(10, 10)
    # @test @varname(A[iseven.(1:10), 1], true) ⊑ @varname(A[1:10, 1])
    # @test @varname(A[iseven.(1:10), 1], true) ⋣ @varname(A[1:10, 1])
    #
    # # we can reasonably allow colons on the right side ("universal set")
    # @test @varname(x[1]) ⊑ @varname(x[:])
    # @test @varname(x[1:10, 1]) ⊑ @varname(x[:, 1:10])
    # @test_throws ErrorException (@varname(x[:]) ⊑ @varname(x[1]))
    # @test_throws ErrorException (@varname(x[:]) ⊑ @varname(x[:]))
    #
    # TODO dynamic indices
    #
    # TODO keyword indices
end

end # module
