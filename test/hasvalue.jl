@testset "base getvalue + hasvalue" begin
    @testset "NamedTuple" begin
        nt = (a=[1], b=2, c=(x=3,), d=[1.0 0.5; 0.5 1.0])
        @test hasvalue(nt, @varname(a))
        @test getvalue(nt, @varname(a)) == [1]
        @test hasvalue(nt, @varname(a[1]))
        @test getvalue(nt, @varname(a[1])) == 1
        @test hasvalue(nt, @varname(b))
        @test getvalue(nt, @varname(b)) == 2
        @test hasvalue(nt, @varname(c))
        @test getvalue(nt, @varname(c)) == (x=3,)
        @test hasvalue(nt, @varname(c.x))
        @test getvalue(nt, @varname(c.x)) == 3
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
        @test !hasvalue(nt, @varname(c.y))
        @test !hasvalue(nt, @varname(d[1, 3]))
        @test !hasvalue(nt, @varname(d[3, :]))
    end

    @testset "Dict" begin
        # same tests as above
        d = Dict(
            @varname(a) => [1],
            @varname(b) => 2,
            @varname(c) => (x=3,),
            @varname(d) => [1.0 0.5; 0.5 1.0],
        )
        @test hasvalue(d, @varname(a))
        @test getvalue(d, @varname(a)) == [1]
        @test hasvalue(d, @varname(a[1]))
        @test getvalue(d, @varname(a[1])) == 1
        @test hasvalue(d, @varname(b))
        @test getvalue(d, @varname(b)) == 2
        @test hasvalue(d, @varname(c))
        @test getvalue(d, @varname(c)) == (x=3,)
        @test hasvalue(d, @varname(c.x))
        @test getvalue(d, @varname(c.x)) == 3
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
        @test !hasvalue(d, @varname(c.y))
        @test !hasvalue(d, @varname(d[1, 3]))

        # extra ones since Dict can have weird keys
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
end

@testset "with Distributions: getvalue + hasvalue" begin
    using Distributions
    using LinearAlgebra

    d = Dict(@varname(x[1]) => 1.0, @varname(x[2]) => 2.0)
    @test hasvalue(d, @varname(x), MvNormal(zeros(2), I))
    @test !hasvalue(d, @varname(x), MvNormal(zeros(3), I))
end
