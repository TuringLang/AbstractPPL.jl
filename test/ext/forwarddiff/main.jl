using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL:
    AbstractPPL, prepare, generate_testcases, run_testcase, value_and_gradient!!
using ADTypes: AutoForwardDiff
using ForwardDiff
using Test

@testset "AbstractPPLForwardDiffExt" begin
    # Julia 1.10 heap-allocates closure captures the 1.11+ runtime elides; mark
    # allocations broken on min so the regression check stays honest on latest.
    alloc_state = VERSION < v"1.11" ? :broken : :test

    @testset "ForwardDiff (default chunk)" begin
        for case in generate_testcases(Val(:vector))
            run_testcase(
                case;
                adtype=AutoForwardDiff(),
                atol=1e-6,
                rtol=1e-6,
                allocations=alloc_state,
                type_stability=:test,
            )
        end
    end

    # `chunksize=2` needs x with at least two elements; skip the `:context`
    # case (x of length 1) and `:edge` cases (chunk doesn't apply).
    @testset "ForwardDiff (explicit chunk)" begin
        ad = AutoForwardDiff(; chunksize=2)
        for case in generate_testcases(Val(:vector))
            case.tag ∈ (:vector, :cache_reuse, :hessian) || continue
            run_testcase(case; adtype=ad, atol=1e-6, rtol=1e-6)
        end
    end

    # `AutoForwardDiff(; tag=...)` exists for nested differentiation. The tag's
    # type parameter is a sentinel chosen by the caller (e.g. DynamicPPL's
    # `DynamicPPLTag`); it intentionally does not equal `typeof(target)`, so
    # the hot path must skip `ForwardDiff.checktag` to avoid a false error.
    @testset "custom AutoForwardDiff tag" begin
        struct OuterTag end
        custom = ForwardDiff.Tag{OuterTag,Float64}()
        x = [1.0, 2.0]
        prep = prepare(AutoForwardDiff(; tag=custom), x -> sum(abs2, x), x)
        @test typeof(prep.cache.config).parameters[1] === typeof(custom)
        val, grad = value_and_gradient!!(prep, x)
        @test val ≈ 5.0
        @test grad ≈ [2.0, 4.0]
    end
end
