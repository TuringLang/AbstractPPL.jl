module VarNameTests

using AbstractPPL
using Test
using JET: @test_call

@testset "varname/varname.jl" verbose = true begin
    @testset "basic construction (and type stability)" begin
        @test @varname(x) == (@inferred VarName{:x}(Iden()))
        @test @varname(x[1]) == (@inferred VarName{:x}(Index((1,), (;), Iden())))
        @test @varname(x.a) == (@inferred VarName{:x}(Property{:a}(Iden())))
        @test @varname(x.a[1]) ==
            (@inferred VarName{:x}(Property{:a}(Index((1,), (;), Iden()))))
    end

    @testset "errors on invalid inputs" begin
        # Note: have to wrap in eval to avoid throwing an error before the actual test
        errmsg = "malformed variable name"
        @test_throws errmsg eval(:(@varname(1)))
        @test_throws errmsg eval(:(@varname(x + y)))
        @test_throws MethodError eval(:(@varname(x[1:Colon()])))
    end

    @testset "equality and hash" begin
        function check_doubleeq_and_hash(vn1, vn2, is_equal)
            if is_equal
                @test vn1 == vn2
                @test hash(vn1) == hash(vn2)
            else
                @test vn1 != vn2
                @test hash(vn1) != hash(vn2)
            end
        end
        check_doubleeq_and_hash(@varname(x), @varname(x), true)
        check_doubleeq_and_hash(@varname(x), @varname(y), false)
        check_doubleeq_and_hash(@varname(x[1]), @varname(x[1]), true)
        check_doubleeq_and_hash(@varname(x[1]), @varname(x[2]), false)
        check_doubleeq_and_hash(@varname(x.a), @varname(x.a), true)
        check_doubleeq_and_hash(@varname(x.a), @varname(x.b), false)
        check_doubleeq_and_hash(@varname(x.a[1]), @varname(x.a[1]), true)
        check_doubleeq_and_hash(@varname(x.a[1]), @varname(x.a[2]), false)
        check_doubleeq_and_hash(@varname(x.a[1]), @varname(x.b[1]), false)
        check_doubleeq_and_hash(@varname(x[1, i=2]), @varname(x[1, i=2]), true)
        check_doubleeq_and_hash(@varname(x[i=2, 4]), @varname(x[4, i=2]), true)

        @testset "dynamic indices" begin
            check_doubleeq_and_hash(@varname(x[begin]), @varname(x[begin]), true)
            check_doubleeq_and_hash(@varname(x[end]), @varname(x[end]), true)
            check_doubleeq_and_hash(@varname(x[begin]), @varname(x[end]), false)
            check_doubleeq_and_hash(@varname(x[begin + 1]), @varname(x[begin + 1]), true)
            check_doubleeq_and_hash(@varname(x[begin + 1]), @varname(x[(begin + 1)]), true)
            check_doubleeq_and_hash(@varname(x[begin + 1]), @varname(x[begin + 2]), false)
            check_doubleeq_and_hash(@varname(x[end - 1]), @varname(x[end - 1]), true)
            check_doubleeq_and_hash(@varname(x[end - 1]), @varname(x[end - 2]), false)
            check_doubleeq_and_hash(
                @varname(x[(begin * end - begin):end]),
                @varname(x[((begin * end) - begin):end]),
                true,
            )
        end
    end

    @testset "JET on equality + dynamic dispatch" begin
        # This test is very specific, so some context is needed:
        #
        # In DynamicPPL it's quite common to want to search for a VarName in a collection of
        # VarNames. Usually the collection will not have a concrete element type (because
        # it's a mixture of different optics). Thus, there will be a fair amount of dynamic
        # dispatch when performing the comparisons.
        #
        # In AbstractPPL, there is custom code in `src/varname/optic.jl` to make sure that
        # equality comparisons of `Index` lenses are JET-friendly even when this happens
        # (i.e., JET.jl doesn't error on `@report_call`). These were needed because base
        # Julia's equality methods on tuples error with JET:
        # https://github.com/JuliaLang/julia/issues/60470, and using those default methods
        # would cause test failures in DynamicPPLJETExt.
        #
        # This test therefore makes sure we don't cause any regressions.
        vns = [@varname(x), @varname(x[1]), @varname(x.a)]
        for vn in vns
            @test_call any(k -> k == vn, vns)
        end
    end

    @testset "pretty-printing" begin
        @test string(@varname(x)) == "x"
        @test string(@varname(x[1])) == "x[1]"
        @test string(@varname(x.a)) == "x.a"
        @test string(@varname(x.a[1])) == "x.a[1]"
        @test string(@varname(x[begin])) == "x[DynamicIndex(begin)]"
        @test string(@varname(x[end])) == "x[DynamicIndex(end)]"
        @test string(@varname(x[:])) == "x[:]"
        @test string(@varname(x[1, i=3])) == "x[1, i=3]"
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
            @test concretize(vn, arr) == @varname(x[1:16])
        end

        @testset "nested" begin
            vn = @varname(x.a[end].b)
            @test vn isa VarName
            @test is_dynamic(vn)
            @test concretize(vn, (; a=[(; b=1)])) == @varname(x.a[1].b)
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
        @test !is_dynamic(@varname(x[k=i]))
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
            @test @varname(x[idx]) == @varname(x[3])
            @test @varname(x[2 * idx]) == @varname(x[6])
            @test @varname(x[1:idx]) == @varname(x[1:3])
            @test @varname(x[k=idx]) == @varname(x[k=3])
            @test @varname(x[k=2 * idx]) == @varname(x[k=6])
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
