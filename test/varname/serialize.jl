module VarNameSerialisationTests

using AbstractPPL
using InvertedIndices: Not, InvertedIndex
using Test

@testset "varname/serialize.jl" verbose = true begin
    @testset "roundtrip" begin
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
            @varname(y[begin:end], true),
            @varname(y[end], true),
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
            @varname(x[i=1]),
            @varname(x[].a[j=2].b[3, 4, 5, [6]]),
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

    @testset "deserialisation fails for unconcretised dynamic indices" begin
        for vn in (@varname(x[1:end]), @varname(x[begin:end]), @varname(x[2:step:end]))
            @test_throws ArgumentError varname_to_string(vn)
        end
    end

    @testset "custom index types" begin
        vn = @varname(x[Not(3)])

        # This won't work as we don't yet know how to handle OffsetArray
        @test_throws MethodError varname_to_string(vn)

        # Now define the relevant methods
        AbstractPPL.index_to_dict(o::InvertedIndex{I}) where {I} = Dict(
            "type" => "InvertedIndices.InvertedIndex",
            "skip" => AbstractPPL.index_to_dict(o.skip),
        )
        AbstractPPL.dict_to_index(::Val{Symbol("InvertedIndices.InvertedIndex")}, d) =
            InvertedIndex(AbstractPPL.dict_to_index(d["skip"]))

        # Serialisation should now work
        @test string_to_varname(varname_to_string(vn)) == vn
    end
end

end # module
