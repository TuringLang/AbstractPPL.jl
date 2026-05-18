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

    # `DICache`'s `Mode` parameter is either `:closure` (compiled-tape
    # ReverseDiff) or the integer context length on the constants path. The
    # constants-path integer also documents how many `DI.Constant`s the AD
    # call passes.
    @testset "DICache encodes the call mode as a type parameter" begin
        x = [1.0, 2.0, 3.0]
        prep_noctx = prepare(AutoForwardDiff(), quadratic, x)
        prep_closure = prepare(AutoReverseDiff(; compile=true), quadratic, x)
        affine(y, a, b) = a * sum(abs2, y) + b
        prep_ctx = prepare(AutoForwardDiff(), affine, x; context=(2.0, 1.0))

        @test prep_noctx.cache isa DIExt.DICache{0}
        @test prep_closure.cache isa DIExt.DICache{:closure}
        @test prep_ctx.cache isa DIExt.DICache{2}

        # Non-empty-context primal matches the underlying `f(x, context...)`.
        @test prep_ctx(x) == affine(x, 2.0, 1.0)
        val, grad = value_and_gradient!!(prep_ctx, x)
        @test val == affine(x, 2.0, 1.0)
        @test grad ≈ [4.0, 8.0, 12.0]  # 2 * 2x

        # Hot path is type-stable on all three preps.
        @inferred value_and_gradient!!(prep_noctx, x)
        @inferred value_and_gradient!!(prep_closure, x)
        @inferred value_and_gradient!!(prep_ctx, x)
    end
end
