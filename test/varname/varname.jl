module VarNameTests

using AbstractPPL
using Test

@testset "varname/varname.jl" verbose = true begin
    @testset "basic construction" begin
        @test @varname(x) == VarName{:x}(Iden())
        @test @varname(x[1]) == VarName{:x}(Index((1,), Iden()))
        @test @varname(x.a) == VarName{:x}(Property{:a}(Iden()))
        @test @varname(x.a[1]) == VarName{:x}(Property{:a}(Index((1,), Iden())))
    end

    @testset "errors on nonsensical inputs" begin
        # Note: have to wrap in eval to avoid throwing an error before the actual test
        errmsg = "malformed variable name"
        @test_throws errmsg eval(:(@varname(1)))
        @test_throws errmsg eval(:(@varname(x + y)))
        # TODO(penelopeysm): I would like to test this, but JuliaFormatter reformats
        # this into x[1::] which then fails to parse. Grr.
        # @test_throws MethodError eval(:(@varname(x[1: :])))
    end

    @testset "dynamic indices and manual concretization" begin
        @testset "begin" begin
            vn = @varname(x[begin])
            @test vn isa VarName
            @test is_dynamic(vn)
            @test concretize(vn, [1.0]) == @varname(x[1])
        end

        @testset "end" begin
            vn = @varname(x[end])
            @test vn isa VarName
            @test is_dynamic(vn)
            @test concretize(vn, randn(5)) == @varname(x[5])
        end

        @testset "expressions thereof" begin
            vn = @varname(x[(begin + 2):(end - 1)])
            @test vn isa VarName
            @test is_dynamic(vn)
            arr = randn(6)
            @test concretize(vn, arr) == @varname(x[3:5])
        end

        @testset "different dimensions" begin
            vn = @varname(x[end, begin:(end - 1)])
            @test vn isa VarName
            @test is_dynamic(vn)
            arr = randn(4, 4)
            @test concretize(vn, arr) == @varname(x[4, 1:3])
        end

        @testset "linear indexing for matrices" begin
            vn = @varname(x[begin:end])
            @test vn isa VarName
            @test is_dynamic(vn)
            arr = randn(4, 4)
            @test concretize(vn, arr) == @varname(x[:])
        end
    end

    @testset "things that shouldn't be dynamic aren't dynamic" begin
        @test !is_dynamic(@varname(x))
        @test !is_dynamic(@varname(x[3]))
        @test !is_dynamic(@varname(x[:]))
        @test !is_dynamic(@varname(x[1:3]))
        @test !is_dynamic(@varname(x[1:3, 3, 2 + 9]))
        i = 10
        @test !is_dynamic(@varname(x[1:3, 3, 2 + 9, 1:3:i]))
    end

    @testset "automatic concretization" begin
        test_array = randn(5, 5)
        @testset "begin" begin
            vn = @varname(test_array[begin], true)
            @test vn == concretize(@varname(test_array[begin]), test_array)
        end
        @testset "end" begin
            vn = @varname(test_array[end], true)
            @test vn == concretize(@varname(test_array[end]), test_array)
        end
        @testset "expressions thereof" begin
            vn = @varname(test_array[(begin + 1):(end - 2)], true)
            @test vn == concretize(@varname(test_array[(begin + 1):(end - 2)]), test_array)
        end
        @testset "different dimensions" begin
            vn = @varname(test_array[end, begin:(end - 1)], true)
            @test vn == concretize(@varname(test_array[end, begin:(end - 1)]), test_array)
        end
        @testset "linear indexing for matrices" begin
            vn = @varname(test_array[begin:end], true)
            @test vn == concretize(@varname(test_array[begin:end]), test_array)
        end
    end
end

end # module VarNameTests
