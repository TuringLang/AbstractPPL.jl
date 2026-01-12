module VarNamePrefixTests

using Test
using AbstractPPL

@testset "varname/prefix.jl" verbose = true begin
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

    @testset "round-trip + type stability" begin
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
                prefixed = @inferred prefix(vn1, vn2)
                @test subsumes(vn2, prefixed)
                unprefixed = @inferred unprefix(prefixed, vn2)
                @test unprefixed == vn1
            end
        end
    end
end

end # module
