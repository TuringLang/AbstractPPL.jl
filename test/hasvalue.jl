@testset "hasvalue" begin
    @testset "NamedTuple" begin
        nt = (a=[1], b=2, c=(x=3,), d=[1.0 0.5; 0.5 1.0])
        @test hasvalue(nt, @varname(a))
        @test hasvalue(nt, @varname(a[1]))
        @test hasvalue(nt, @varname(b))
        @test hasvalue(nt, @varname(c))
        @test hasvalue(nt, @varname(c.x))
        @test hasvalue(nt, @varname(d))
        @test hasvalue(nt, @varname(d[1, 1]))
        @test hasvalue(nt, @varname(d[1, 2]))
        @test hasvalue(nt, @varname(d[2, 1]))
        @test hasvalue(nt, @varname(d[2, 2]))
        @test hasvalue(nt, @varname(d[3]))  # linear indexing works....
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
        d = Dict(@varname(a) => [1],
            @varname(b) => 2,
            @varname(c) => (x=3,),
            @varname(d) => [1.0 0.5; 0.5 1.0])
        @test hasvalue(d, @varname(a))
        @test hasvalue(d, @varname(a[1]))
        @test hasvalue(d, @varname(b))
        @test hasvalue(d, @varname(c))
        @test hasvalue(d, @varname(c.x))
        @test hasvalue(d, @varname(d))
        @test hasvalue(d, @varname(d[1, 1]))
        @test hasvalue(d, @varname(d[1, 2]))
        @test hasvalue(d, @varname(d[2, 1]))
        @test hasvalue(d, @varname(d[2, 2]))
        @test hasvalue(d, @varname(d[3])) # linear indexing works....
        @test !hasvalue(d, @varname(nope))
        @test !hasvalue(d, @varname(a[2]))
        @test !hasvalue(d, @varname(a[1][1]))
        @test !hasvalue(d, @varname(c.x[1]))
        @test !hasvalue(d, @varname(c.y))
        @test !hasvalue(d, @varname(d[1, 3]))
        @test !hasvalue(d, @varname(d[3]))

        # extra ones since Dict can have weird key
        d = Dict(@varname(a[1]) => [1.0, 2.0],
                 @varname(b.x) => [3.0])
        @test hasvalue(d, @varname(a[1]))
        @test hasvalue(d, @varname(a[1][1]))
        @test hasvalue(d, @varname(a[1][2]))
        @test hasvalue(d, @varname(b.x))
        @test hasvalue(d, @varname(b.x[1]))
        @test !hasvalue(d, @varname(a))
        @test !hasvalue(d, @varname(a[2]))
        @test !hasvalue(d, @varname(b.y))
        @test !hasvalue(d, @varname(b.x[2]))
    end
end
