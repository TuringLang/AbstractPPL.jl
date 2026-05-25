using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: AbstractPPL, prepare, run_testcases, value_and_gradient!!
using ADTypes: AutoForwardDiff
using ForwardDiff
using Test

@testset "AbstractPPLForwardDiffExt" begin
    @testset "ForwardDiff (default chunk)" begin
        run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:hessian); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=AutoForwardDiff())
        # Julia 1.10 heap-allocates some `Fix2`/closure captures that 1.11+
        # elides. Mark `:allocations` broken on min to flag the regression
        # detection without failing the suite on the older runtime.
        run_testcases(
            Val(:allocations);
            adtype=AutoForwardDiff(),
            gradient_broken=VERSION < v"1.11",
            jacobian_broken=VERSION < v"1.11",
        )
        run_testcases(Val(:type_stability); adtype=AutoForwardDiff())
    end

    @testset "ForwardDiff (explicit chunk)" begin
        run_testcases(
            Val(:vector); adtype=AutoForwardDiff(; chunksize=2), atol=1e-6, rtol=1e-6
        )
        run_testcases(
            Val(:cache_reuse); adtype=AutoForwardDiff(; chunksize=2), atol=1e-6, rtol=1e-6
        )
    end

    @testset "AutoForwardDiff context" begin
        run_testcases(Val(:context); adtype=AutoForwardDiff(), atol=1e-10, rtol=1e-10)
    end

    # `AutoForwardDiff(; tag=...)` exists for nested differentiation. Check the
    # user-supplied tag is threaded into the ForwardDiff config (the inner
    # `*Config` carries the tag in its first type parameter).
    @testset "custom AutoForwardDiff tag" begin
        struct OuterTag end
        custom = ForwardDiff.Tag{OuterTag,Float64}()
        prep = prepare(AutoForwardDiff(; tag=custom), x -> sum(abs2, x), [1.0, 2.0])
        @test typeof(prep.cache.config).parameters[1] === typeof(custom)
    end
end
