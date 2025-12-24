using Accessors
using InvertedIndices
using OffsetArrays

@testset "varnames" begin
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
end
