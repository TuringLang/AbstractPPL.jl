@testset "base getvalue + hasvalue" begin
    @testset "basic NamedTuple" begin
        nt = (a=[1], b=2, c=(x=3, y=[4], z=(; p=[(; q=5)])), d=[1.0 0.5; 0.5 1.0])
        @test hasvalue(nt, @varname(a))
        @test getvalue(nt, @varname(a)) == [1]
        @test hasvalue(nt, @varname(a[1]))
        @test getvalue(nt, @varname(a[1])) == 1
        @test hasvalue(nt, @varname(b))
        @test getvalue(nt, @varname(b)) == 2
        @test hasvalue(nt, @varname(c))
        @test getvalue(nt, @varname(c)) == (x=3, y=[4], z=(p = [(; q=5)]))
        @test hasvalue(nt, @varname(c.x))
        @test getvalue(nt, @varname(c.x)) == 3
        @test hasvalue(nt, @varname(c.y))
        @test getvalue(nt, @varname(c.y)) == [4]
        @test hasvalue(nt, @varname(c.y[1]))
        @test getvalue(nt, @varname(c.y[1])) == 4
        @test hasvalue(nt, @varname(c.z))
        @test getvalue(nt, @varname(c.z)) == (; p=[(; q=5)])
        @test hasvalue(nt, @varname(c.z.p))
        @test getvalue(nt, @varname(c.z.p)) == [(; q=5)]
        @test hasvalue(nt, @varname(c.z.p[1]))
        @test getvalue(nt, @varname(c.z.p[1])) == (; q=5)
        @test hasvalue(nt, @varname(c.z.p[1].q))
        @test getvalue(nt, @varname(c.z.p[1].q)) == 5
        @test hasvalue(nt, @varname(d))
        @test getvalue(nt, @varname(d)) == [1.0 0.5; 0.5 1.0]
        @test hasvalue(nt, @varname(d[1, 1]))
        @test getvalue(nt, @varname(d[1, 1])) == 1.0
        @test hasvalue(nt, @varname(d[1, 2]))
        @test getvalue(nt, @varname(d[1, 2])) == 0.5
        @test hasvalue(nt, @varname(d[2, 1]))
        @test getvalue(nt, @varname(d[2, 1])) == 0.5
        @test hasvalue(nt, @varname(d[2, 2]))
        @test getvalue(nt, @varname(d[2, 2])) == 1.0
        @test hasvalue(nt, @varname(d[3]))  # linear indexing works....
        @test getvalue(nt, @varname(d[3])) == 0.5
        @test !hasvalue(nt, @varname(nope))
        @test !hasvalue(nt, @varname(a[2]))
        @test !hasvalue(nt, @varname(a[1][1]))
        @test !hasvalue(nt, @varname(c.x[1]))
        @test !hasvalue(nt, @varname(c.y[2]))
        @test !hasvalue(nt, @varname(c.y.a))
        @test !hasvalue(nt, @varname(c.zzzz))
        @test !hasvalue(nt, @varname(d[1, 3]))
        @test !hasvalue(nt, @varname(d[3, :]))
    end

    @testset "basic Dict" begin
        # same tests as for NamedTuple
        d = Dict(
            @varname(a) => [1],
            @varname(b) => 2,
            @varname(c) => (x=3, y=[4], z=(; p=[(; q=5)])),
            @varname(d) => [1.0 0.5; 0.5 1.0],
        )
        @test hasvalue(d, @varname(a))
        @test getvalue(d, @varname(a)) == [1]
        @test hasvalue(d, @varname(a[1]))
        @test getvalue(d, @varname(a[1])) == 1
        @test hasvalue(d, @varname(b))
        @test getvalue(d, @varname(b)) == 2
        @test hasvalue(d, @varname(c))
        @test getvalue(d, @varname(c)) == (x=3, y=[4], z=(p = [(; q=5)]))
        @test hasvalue(d, @varname(c.x))
        @test getvalue(d, @varname(c.x)) == 3
        @test hasvalue(d, @varname(c.y))
        @test getvalue(d, @varname(c.y)) == [4]
        @test hasvalue(d, @varname(c.y[1]))
        @test getvalue(d, @varname(c.y[1])) == 4
        @test hasvalue(d, @varname(c.z))
        @test getvalue(d, @varname(c.z)) == (; p=[(; q=5)])
        @test hasvalue(d, @varname(c.z.p))
        @test getvalue(d, @varname(c.z.p)) == [(; q=5)]
        @test hasvalue(d, @varname(c.z.p[1]))
        @test getvalue(d, @varname(c.z.p[1])) == (; q=5)
        @test hasvalue(d, @varname(c.z.p[1].q))
        @test getvalue(d, @varname(c.z.p[1].q)) == 5
        @test hasvalue(d, @varname(d))
        @test getvalue(d, @varname(d)) == [1.0 0.5; 0.5 1.0]
        @test hasvalue(d, @varname(d[1, 1]))
        @test getvalue(d, @varname(d[1, 1])) == 1.0
        @test hasvalue(d, @varname(d[1, 2]))
        @test getvalue(d, @varname(d[1, 2])) == 0.5
        @test hasvalue(d, @varname(d[2, 1]))
        @test getvalue(d, @varname(d[2, 1])) == 0.5
        @test hasvalue(d, @varname(d[2, 2]))
        @test getvalue(d, @varname(d[2, 2])) == 1.0
        @test hasvalue(d, @varname(d[3]))  # linear indexing works....
        @test getvalue(d, @varname(d[3])) == 0.5
        @test !hasvalue(d, @varname(nope))
        @test !hasvalue(d, @varname(a[2]))
        @test !hasvalue(d, @varname(a[1][1]))
        @test !hasvalue(d, @varname(c.x[1]))
        @test !hasvalue(d, @varname(c.y[2]))
        @test !hasvalue(d, @varname(c.y.a))
        @test !hasvalue(d, @varname(c.zzzz))
        @test !hasvalue(d, @varname(d[1, 3]))
    end

    @testset "Dict with non-identity varname keys" begin
        d = Dict(
            @varname(a[1]) => [1.0, 2.0],
            @varname(b.x) => [3.0],
            @varname(c[2]) => (a=4.0, b=5.0),
        )
        @test hasvalue(d, @varname(a[1]))
        @test getvalue(d, @varname(a[1])) == [1.0, 2.0]
        @test hasvalue(d, @varname(a[1][1]))
        @test getvalue(d, @varname(a[1][1])) == 1.0
        @test hasvalue(d, @varname(a[1][2]))
        @test getvalue(d, @varname(a[1][2])) == 2.0
        @test hasvalue(d, @varname(b.x))
        @test getvalue(d, @varname(b.x)) == [3.0]
        @test hasvalue(d, @varname(b.x[1]))
        @test getvalue(d, @varname(b.x[1])) == 3.0
        @test hasvalue(d, @varname(c[2]))
        @test getvalue(d, @varname(c[2])) == (a=4.0, b=5.0)
        @test hasvalue(d, @varname(c[2].a))
        @test getvalue(d, @varname(c[2].a)) == 4.0
        @test hasvalue(d, @varname(c[2].b))
        @test getvalue(d, @varname(c[2].b)) == 5.0
        @test !hasvalue(d, @varname(a))
        @test !hasvalue(d, @varname(a[2]))
        @test !hasvalue(d, @varname(b.y))
        @test !hasvalue(d, @varname(b.x[2]))
        @test !hasvalue(d, @varname(c[1]))
        @test !hasvalue(d, @varname(c[2].x))
    end

    @testset "Dict with redundancy" begin
        d1 = Dict(@varname(x) => [[[[1.0]]]])
        d2 = Dict(@varname(x[1]) => [[[2.0]]])
        d3 = Dict(@varname(x[1][1]) => [[3.0]])
        d4 = Dict(@varname(x[1][1][1]) => [4.0])
        d5 = Dict(@varname(x[1][1][1][1]) => 5.0)

        d = Dict{VarName,Any}()
        for (new_dict, expected_value) in
            zip((d1, d2, d3, d4, d5), (1.0, 2.0, 3.0, 4.0, 5.0))
            d = merge(d, new_dict)
            @test hasvalue(d, @varname(x[1][1][1][1]))
            @test getvalue(d, @varname(x[1][1][1][1])) == expected_value
            # for good measure
            @test !hasvalue(d, @varname(x[1][1][1][2]))
            @test !hasvalue(d, @varname(x[1][1][2][1]))
            @test !hasvalue(d, @varname(x[1][2][1][1]))
            @test !hasvalue(d, @varname(x[2][1][1][1]))
        end
    end
end

@testset "with Distributions: getvalue + hasvalue" begin
    using Distributions
    using LinearAlgebra

    @testset "univariate" begin
        d = Dict(@varname(x) => 1.0, @varname(y) => [[2.0]])
        @test hasvalue(d, @varname(x), Normal())
        @test getvalue(d, @varname(x), Normal()) == 1.0
        @test hasvalue(d, @varname(y[1][1]), Normal())
        @test getvalue(d, @varname(y[1][1]), Normal()) == 2.0
    end

    @testset "multivariate + matrix" begin
        d = Dict(@varname(x[1]) => 1.0, @varname(x[2]) => 2.0)
        @test hasvalue(d, @varname(x), MvNormal(zeros(1), I))
        @test getvalue(d, @varname(x), MvNormal(zeros(1), I)) == [1.0]
        @test hasvalue(d, @varname(x), MvNormal(zeros(2), I))
        @test getvalue(d, @varname(x), MvNormal(zeros(2), I)) == [1.0, 2.0]
        @test !hasvalue(d, @varname(x), MvNormal(zeros(3), I))
        @test_throws ErrorException hasvalue(
            d, @varname(x), MvNormal(zeros(3), I); error_on_incomplete=true
        )
        # If none of the varnames match, it should just return false instead of erroring
        @test !hasvalue(d, @varname(y), MvNormal(zeros(2), I); error_on_incomplete=true)
    end

    @testset "LKJCholesky :upside_down_smile:" begin
        # yes, this isn't a valid Cholesky sample, but whatever
        d = Dict(
            @varname(x.L[1, 1]) => 1.0,
            @varname(x.L[2, 1]) => 2.0,
            @varname(x.L[2, 2]) => 3.0,
        )
        @test hasvalue(d, @varname(x), LKJCholesky(2, 1.0))
        @test getvalue(d, @varname(x), LKJCholesky(2, 1.0)) ==
            Cholesky(LowerTriangular([1.0 0.0; 2.0 3.0]))
        @test !hasvalue(d, @varname(x), LKJCholesky(3, 1.0))
        @test_throws ErrorException hasvalue(
            d, @varname(x), LKJCholesky(3, 1.0); error_on_incomplete=true
        )
        @test !hasvalue(d, @varname(y), LKJCholesky(3, 1.0); error_on_incomplete=true)

        d = Dict(
            @varname(x.U[1, 1]) => 1.0,
            @varname(x.U[1, 2]) => 2.0,
            @varname(x.U[2, 2]) => 3.0,
        )
        @test hasvalue(d, @varname(x), LKJCholesky(2, 1.0, :U))
        @test getvalue(d, @varname(x), LKJCholesky(2, 1.0, :U)) ==
            Cholesky(UpperTriangular([1.0 2.0; 0.0 3.0]))
        @test !hasvalue(d, @varname(x), LKJCholesky(3, 1.0, :U))
        @test_throws ErrorException hasvalue(
            d, @varname(x), LKJCholesky(3, 1.0, :U); error_on_incomplete=true
        )
        @test !hasvalue(d, @varname(y), LKJCholesky(3, 1.0, :U); error_on_incomplete=true)
    end
end
