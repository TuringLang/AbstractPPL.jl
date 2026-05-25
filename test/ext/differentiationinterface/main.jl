using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL:
    AbstractPPL,
    prepare,
    generate_testcases,
    run_testcase,
    value_and_gradient!!,
    value_gradient_and_hessian!!,
    order
using ADTypes: AutoForwardDiff, AutoReverseDiff
using DifferentiationInterface: DifferentiationInterface as DI, SecondOrder
using ForwardDiff
using ReverseDiff
using Test

const DIExt = Base.get_extension(AbstractPPL, :AbstractPPLDifferentiationInterfaceExt)

quadratic(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    @testset "ForwardDiff" begin
        for case in generate_testcases(Val(:vector))
            run_testcase(case; adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        end
    end

    # Compiled-tape ReverseDiff closes the evaluator into a `Base.Fix2` target
    # via `_di_call_shape(::AutoReverseDiff{true}, …)`; the `:cache_reuse`
    # cases exercise that path across multiple inputs. Skip `:hessian`
    # (compiled tape doesn't support `prepare_hessian`).
    @testset "ReverseDiff (compiled tape)" begin
        adtype = AutoReverseDiff(; compile=true)
        for case in generate_testcases(Val(:vector))
            case.tag === :hessian && continue
            run_testcase(case; adtype, atol=1e-6, rtol=1e-6)
        end
    end

    # The DI cache types' `Mode` parameter is either `:closure` (compiled-tape
    # ReverseDiff) or the integer context length on the constants path. The
    # constants-path integer also documents how many `DI.Constant`s the AD
    # call passes. `AutoReverseDiff()` (non-compiled) is used here because the
    # direct `AbstractPPLForwardDiffExt` path takes precedence over DI for
    # `AutoForwardDiff` when both extensions are loaded.
    @testset "DI cache encodes the call mode as a type parameter" begin
        x = [1.0, 2.0, 3.0]
        prep_noctx = prepare(AutoReverseDiff(), quadratic, x)
        prep_closure = prepare(AutoReverseDiff(; compile=true), quadratic, x)
        affine(y, a, b) = a * sum(abs2, y) + b
        prep_ctx = prepare(AutoReverseDiff(), affine, x; context=(2.0, 1.0))

        @test prep_noctx.cache isa DIExt.DIGradientCache{0}
        @test prep_closure.cache isa DIExt.DIGradientCache{:closure}
        @test prep_ctx.cache isa DIExt.DIGradientCache{2}

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

    # `SecondOrder(outer, inner)` lets the caller pick the inner gradient
    # backend and the outer differentiator independently — useful when the
    # default Hessian strategy DI picks for a single `adtype` is suboptimal.
    # Since `SecondOrder <: AbstractADType`, the existing `order=2` dispatch
    # routes it through `DI.prepare_hessian` / `DI.value_gradient_and_hessian`
    # without any extension-side changes.
    @testset "SecondOrder for order=2" begin
        adtype = SecondOrder(AutoForwardDiff(), AutoReverseDiff())
        x = [1.0, 2.0, 3.0]
        prep = prepare(adtype, quadratic, zeros(3); order=2)
        @test order(prep) == 2
        val, grad, hess = value_gradient_and_hessian!!(prep, x)
        @test val ≈ 14.0
        @test grad ≈ [2.0, 4.0, 6.0]
        @test hess ≈ [2.0 0 0; 0 2.0 0; 0 0 2.0]

        # `value_and_gradient!!` on a `SecondOrder` order=2 prep routes through
        # the inner adtype — the only non-trivial case for `_gradient_adtype`.
        val1, grad1 = value_and_gradient!!(prep, x)
        @test val1 ≈ 14.0
        @test grad1 ≈ [2.0, 4.0, 6.0]

        # `context=` composes with `SecondOrder` the same way as for a plain `adtype`.
        affine(y, a, b) = a * sum(abs2, y) + b
        prep_ctx = prepare(adtype, affine, zeros(3); context=(2.0, 1.0), order=2)
        val_ctx, grad_ctx, hess_ctx = value_gradient_and_hessian!!(prep_ctx, x)
        @test val_ctx ≈ affine(x, 2.0, 1.0)
        @test grad_ctx ≈ [4.0, 8.0, 12.0]
        @test hess_ctx ≈ [4.0 0 0; 0 4.0 0; 0 0 4.0]
    end
end
