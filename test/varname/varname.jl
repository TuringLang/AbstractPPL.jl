module VarNameTests

using AbstractPPL
using Test

@testset "varname/varname.jl" verbose = true begin
    # TODO

    @testset "errors on nonsensical inputs" begin
        # Note: have to wrap in eval to avoid throwing an error before the actual test
        errmsg = "malformed variable name"
        @test_throws errmsg eval(:(@varname(1)))
        @test_throws errmsg eval(:(@varname(x + y)))
        # This doesn't fail to parse, but it will throw a MethodError because you can't
        # construct a DynamicColon with a DynamicColon as an argument
        # TODO(penelopeysm): I would like to test this, but JuliaFormatter reformats
        # this into x[1::] which then fails to parse. Grr.
        # @test_throws MethodError eval(:(@varname(x[1: :])))
    end
end

end # module VarNameTests
