module VarNameSubsumesTests

using AbstractPPL
using Test

@testset "varname/subsumes.jl" verbose = true begin
    @testset "varnames that are equal" begin
        @test subsumes(@varname(x), @varname(x))
        @test subsumes(@varname(x[1]), @varname(x[1]))
        @test subsumes(@varname(x.a), @varname(x.a))
    end

    uncomparable(vn1, vn2) = !subsumes(vn1, vn2) && !subsumes(vn2, vn1)
    @testset "uncomparable varnames" begin
        @test uncomparable(@varname(x), @varname(y))
        @test uncomparable(@varname(x.a), @varname(y.a))
        @test uncomparable(@varname(a.x), @varname(a.y))
        @test uncomparable(@varname(a.x[1]), @varname(a.x.z))
        @test uncomparable(@varname(x[1]), @varname(y[1]))
        @test uncomparable(@varname(x[1]), @varname(x.y))
    end

    strictly_subsumes(vn1, vn2) = subsumes(vn1, vn2) && !subsumes(vn2, vn1)
    @testset "strict subsumption - no index comparisons" begin
        @test strictly_subsumes(@varname(x), @varname(x.a))
        @test strictly_subsumes(@varname(x), @varname(x[1]))
        @test strictly_subsumes(@varname(x), @varname(x[2:2:5]))
        @test strictly_subsumes(@varname(x), @varname(x[10, 20]))
        @test strictly_subsumes(@varname(x.a), @varname(x.a.b))
        @test strictly_subsumes(@varname(x[1]), @varname(x[1].a))
        @test strictly_subsumes(@varname(x.a), @varname(x.a[1]))
        @test strictly_subsumes(@varname(x[1:10]), @varname(x[1:10][2]))
    end

    @testset "strict subsumption - index comparisons" begin
        @testset "integer vectors" begin
            @test strictly_subsumes(@varname(x[1:10]), @varname(x[1]))
            @test strictly_subsumes(@varname(x[1:10]), @varname(x[1:5]))
            @test strictly_subsumes(@varname(x[1:10]), @varname(x[4:6]))
            @test strictly_subsumes(@varname(x[1:10, 1:10]), @varname(x[1:5, 1:5]))
            @test strictly_subsumes(@varname(x[[5, 4, 3, 2, 1]]), @varname(x[[2, 4]]))
        end

        @testset "non-integer indices" begin
            @test strictly_subsumes(@varname(x[:a]), @varname(x[:a][1]))
        end

        @testset "colon" begin
            @test strictly_subsumes(@varname(x[:]), @varname(x[1]))
            @test strictly_subsumes(@varname(x[:, 1:10]), @varname(x[1:10, 1]))
        end

        @testset "dynamic indices" begin
            @test strictly_subsumes(@varname(x), @varname(x[begin]))
            @test subsumes(@varname(x[begin]), @varname(x[begin]))
            @test strictly_subsumes(@varname(x[:]), @varname(x[begin]))
            @test strictly_subsumes(@varname(x), @varname(x[end]))
            @test subsumes(@varname(x[end]), @varname(x[end]))
            @test strictly_subsumes(@varname(x[:]), @varname(x[end]))
            @test strictly_subsumes(@varname(x[:]), @varname(x[1:end]))
            @test strictly_subsumes(@varname(x[:]), @varname(x[end - 3]))
        end

        @testset "keyword indices" begin
            @test strictly_subsumes(@varname(x), @varname(x[a=1]))
            @test strictly_subsumes(@varname(x[a=1:10, b=1:10]), @varname(x[a=1:10]))
            @test strictly_subsumes(@varname(x[a=1:10, b=1:10]), @varname(x[a=1:5, b=1:5]))
            @test strictly_subsumes(@varname(x[a=:]), @varname(x[a=1]))
            @test uncomparable(@varname(x[a=1:10, b=5]), @varname(x[a=5, b=1:10]))
            @test uncomparable(@varname(x[a=1]), @varname(x[b=1]))
        end
    end
end

end # module
