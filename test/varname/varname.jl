module VarNameTests

using AbstractPPL
using Test

@testset "varname/varname.jl" verbose = true begin
    @testset "basic construction (and type stability)" begin
        @test @varname(x) == (@inferred VarName{:x}(Iden()))
        @test @varname(x[1]) == (@inferred VarName{:x}(Index((1,), Iden())))
        @test @varname(x.a) == (@inferred VarName{:x}(Property{:a}(Iden())))
        @test @varname(x.a[1]) == (@inferred VarName{:x}(Property{:a}(Index((1,), Iden()))))
    end

    @testset "errors on invalid inputs" begin
        # Note: have to wrap in eval to avoid throwing an error before the actual test
        errmsg = "malformed variable name"
        @test_throws errmsg eval(:(@varname(1)))
        @test_throws errmsg eval(:(@varname(x + y)))
        @test_throws MethodError eval(:(@varname(x[1:Colon()])))
    end

    @testset "equality" begin
        @test @varname(x) == @varname(x)
        @test @varname(x) != @varname(y)
        @test @varname(x[1]) == @varname(x[1])
        @test @varname(x[1]) != @varname(x[2])
        @test @varname(x.a) == @varname(x.a)
        @test @varname(x.a) != @varname(x.b)
        @test @varname(x.a[1]) == @varname(x.a[1])
        @test @varname(x.a[1]) != @varname(x.a[2])
        @test @varname(x.a[1]) != @varname(x.b[1])
    end

    @testset "pretty-printing" begin
        @test string(@varname(x)) == "x"
        @test string(@varname(x[1])) == "x[1]"
        @test string(@varname(x.a)) == "x.a"
        @test string(@varname(x.a[1])) == "x.a[1]"
        @test string(@varname(x[begin])) == "x[DynamicIndex(begin)]"
        @test string(@varname(x[end])) == "x[DynamicIndex(end)]"
        @test string(@varname(x[:])) == "x[:]"
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

    @testset "interpolation" begin
        @testset "of property names" begin
            prop = :myprop
            vn = @varname(x.$prop)
            @test vn == @varname(x.myprop)
        end

        @testset "of indices" begin
            idx = 3
            vn = @varname(x[idx])
            @test vn == @varname(x[3])
        end

        @testset "with dynamic indices" begin
            idx = 3
            vn = @varname(x[end - idx])
            @test vn isa VarName
            @test is_dynamic(vn)
            arr = randn(6)
            @test concretize(vn, arr) == @varname(x[3])
            # Note that `idx` is only resolved at concretization time (because it's stored
            # in a function that looks like (val -> lastindex(val) - idx) -- the VALUE of
            # `idx` is not interpolated at macro time because we have no way of obtaining
            # values inside the macro). So we could change it and re-concretize...
            idx = 4
            @test concretize(vn, arr) == @varname(x[2])
        end

        @testset "of top-level name" begin
            name = :x
            @test @varname($name) == @varname(x)
            @test @varname($name[1]) == @varname(x[1])
            @test @varname($name.a) == @varname(x.a)
        end

        @testset "mashup of everything" begin
            name = :x
            index = 2
            prop = :b
            @test @varname($name.$prop[3 * index]) == @varname(x.b[6])
        end
    end
end

end # module VarNameTests
