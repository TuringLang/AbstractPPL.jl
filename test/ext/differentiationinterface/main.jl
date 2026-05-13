using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: AbstractPPL, prepare, run_testcases, value_and_gradient!!
using ADTypes: AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff
using ReverseDiff
using Test

const DIExt = Base.get_extension(AbstractPPL, :AbstractPPLDifferentiationInterfaceExt)

quadratic(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    @testset "ForwardDiff" begin
        run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=AutoForwardDiff())
    end

    # Compiled-tape ReverseDiff goes through the `_prepare_di(::AutoReverseDiff{true}, …)`
    # specialisation that closes the evaluator into a `Base.Fix2` target — the
    # `:cache_reuse` group exercises that path across multiple inputs.
    @testset "ReverseDiff (compiled tape)" begin
        adtype = AutoReverseDiff(; compile=true)
        run_testcases(Val(:vector); adtype=adtype, atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=adtype, atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=adtype)
    end

    # `DICache` encodes `UseContext` as a type parameter so the
    # context-vs-no-context DI call is resolved by dispatch, not a runtime
    # `Bool` branch in the AD hot path.
    @testset "DICache encodes UseContext as a type parameter" begin
        x = [1.0, 2.0, 3.0]
        prep_ctx = prepare(AutoForwardDiff(), quadratic, x)
        prep_noctx = prepare(AutoReverseDiff(; compile=true), quadratic, x)

        @test prep_ctx.cache isa DIExt.DICache{true}
        @test prep_noctx.cache isa DIExt.DICache{false}
        @test !hasfield(typeof(prep_ctx.cache), :use_context)

        # Hot path is type-stable on both branches.
        @inferred value_and_gradient!!(prep_ctx, x)
        @inferred value_and_gradient!!(prep_noctx, x)
    end
end
