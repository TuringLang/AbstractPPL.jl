using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL:
    AbstractPPL,
    prepare,
    run_testcases,
    value_and_gradient!!,
    value_and_jacobian!!,
    value_gradient_and_hessian!!,
    order
using ADTypes: AutoForwardDiff
using ForwardDiff
using DiffResults
using Test

@testset "AbstractPPLForwardDiffExt" begin
    @testset "ForwardDiff (default chunk)" begin
        run_testcases(Val(:vector); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:hessian); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:cache_reuse); adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        run_testcases(Val(:edge); adtype=AutoForwardDiff())
    end

    @testset "ForwardDiff (explicit chunk)" begin
        run_testcases(
            Val(:vector); adtype=AutoForwardDiff(; chunksize=2), atol=1e-6, rtol=1e-6
        )
        run_testcases(
            Val(:cache_reuse); adtype=AutoForwardDiff(; chunksize=2), atol=1e-6, rtol=1e-6
        )
    end

    @testset "context-lowered gradient" begin
        raw_logdensity(x::AbstractVector{<:Real}, offset) = -0.5 * (x[1] - offset)^2

        x = [0.3]
        ad = AutoForwardDiff()

        lowered = prepare(ad, raw_logdensity, x; check_dims=false, context=(0.1,))

        # `prepared(x)` evaluates `raw_logdensity(x, context...)`.
        @test lowered(x) == raw_logdensity(x, 0.1)

        # Gradient differentiates only w.r.t. `x`.
        val, grad = value_and_gradient!!(lowered, x)
        @test val ≈ raw_logdensity(x, 0.1)
        @test grad ≈ [-(x[1] - 0.1)] atol = 1e-10

        # Jacobian on a scalar-only lowered cache surfaces our arity-mismatch error.
        @test_throws r"vector-valued" value_and_jacobian!!(lowered, x)
    end

    @testset "empty input" begin
        # Gradient with zero-length input.
        ad = AutoForwardDiff()
        f_scalar(x::AbstractVector) = sum(x; init=0.0)
        prep = prepare(ad, f_scalar, Float64[])
        val, grad = value_and_gradient!!(prep, Float64[])
        @test val == 0.0
        @test grad == Float64[]

        # Jacobian with zero-length input.
        f_vec(x::AbstractVector) = [sum(x; init=0.0), 1.0]
        prep_j = prepare(ad, f_vec, Float64[])
        val_j, jac = value_and_jacobian!!(prep_j, Float64[])
        @test val_j == [0.0, 1.0]
        @test size(jac) == (2, 0)

        # Hessian with zero-length input.
        prep_h = prepare(ad, f_scalar, Float64[]; order=2)
        @test order(prep_h) == 2
        val_h, grad_h, hess_h = value_gradient_and_hessian!!(prep_h, Float64[])
        @test val_h == 0.0
        @test grad_h == Float64[]
        @test size(hess_h) == (0, 0)
    end
end
