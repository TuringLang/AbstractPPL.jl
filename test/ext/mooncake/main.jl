using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: AbstractPPL, prepare, run_testcases, value_and_gradient!!
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake
using Test

@testset "AbstractPPLMooncakeExt" begin
    for (label, adtype) in (
        ("Mooncake (reverse)", AutoMooncake()),
        ("Mooncake (forward)", AutoMooncakeForward()),
    )
        @testset "$label" begin
            run_testcases(Val(:vector); adtype=adtype, atol=1e-6, rtol=1e-6)
            run_testcases(Val(:namedtuple); adtype=adtype, atol=1e-6, rtol=1e-6)
            run_testcases(Val(:cache_reuse); adtype=adtype, atol=1e-6, rtol=1e-6)
            run_testcases(Val(:edge); adtype=adtype)
            # Hessian (`order=2`) is reverse-mode only on the AutoMooncake side;
            # AutoMooncakeForward routes through the same generic Hessian path
            # since `Mooncake.prepare_hessian_cache` is mode-agnostic.
            run_testcases(Val(:hessian); adtype=adtype, atol=1e-6, rtol=1e-6)
            # Mooncake's `value_and_jacobian!!` currently allocates fresh
            # cotangent/Jacobian buffers each call, and the forward-mode
            # Jacobian return type infers as `Tuple{Any, Union{Array{T,3},
            # Matrix}}`. Mark those known-broken; the other paths must hold.
            run_testcases(Val(:allocations); adtype=adtype, jacobian_broken=true)
            run_testcases(
                Val(:type_stability);
                adtype=adtype,
                jacobian_broken=adtype isa AutoMooncakeForward,
            )
        end
    end

    @testset "context-lowered gradient" begin
        struct TinyProblem{T}
            offset::T
        end
        raw_logdensity(x::AbstractVector{<:Real}, offset) = -0.5 * (x[1] - offset)^2
        (p::TinyProblem)(x::AbstractVector{<:Real}) = raw_logdensity(x, p.offset)

        x = [0.3]
        problem = TinyProblem(0.1)
        ad = AutoMooncake(; config=nothing)

        generic = prepare(ad, problem, x; check_dims=false)
        lowered = prepare(
            ad, raw_logdensity, x; check_dims=false, context=(problem.offset,)
        )

        # `prepared(x)` evaluates `problem(x)` on the generic path and
        # `raw_logdensity(x, context...)` on the lowered path; both should
        # produce the same scalar.
        @test generic(x) == problem(x)
        @test lowered(x) == problem(x)

        # Same value and gradient as the generic path.
        @test value_and_gradient!!(generic, x) == value_and_gradient!!(lowered, x)

        # Forward mode supports context too — same primal and (approximately)
        # the same derivative as the reverse-mode lowered path on this scalar
        # problem. Use `≈` because forward and reverse may differ in the last
        # ULPs.
        ad_fwd = AutoMooncakeForward(; config=nothing)
        lowered_fwd = prepare(
            ad_fwd, raw_logdensity, x; check_dims=false, context=(problem.offset,)
        )
        @test lowered_fwd(x) == problem(x)
        val_fwd, grad_fwd = value_and_gradient!!(lowered_fwd, x)
        val_rev, grad_rev = value_and_gradient!!(lowered, x)
        @test val_fwd ≈ val_rev atol = 1e-12
        @test grad_fwd ≈ grad_rev atol = 1e-12

        # Rejects on vector-valued problems with non-empty context.
        vec_problem(y, c) = [y[1] * c, y[1] + c]
        @test_throws ArgumentError prepare(
            ad, vec_problem, x; check_dims=false, context=(1.0,)
        )

        # Empty input with non-empty context is supported — the empty-input
        # shortcut bypasses Mooncake and just calls `f([], context...)`. Use
        # a `sum(...; init=0.0)`-based `f` since `raw_logdensity` indexes `x[1]`.
        empty_logdensity(y::AbstractVector{<:Real}, offset) =
            sum(y; init=zero(eltype(y))) + offset
        empty_lowered = prepare(
            ad, empty_logdensity, Float64[]; check_dims=false, context=(0.5,)
        )
        val0, grad0 = value_and_gradient!!(empty_lowered, Float64[])
        @test val0 == empty_logdensity(Float64[], 0.5)
        @test grad0 == Float64[]

        # Jacobian on a scalar-only lowered cache surfaces our arity-mismatch error.
        @test_throws r"vector-valued" AbstractPPL.value_and_jacobian!!(lowered, x)
    end

    @testset "dense vector requirement" begin
        # Non-dense AbstractVectors (e.g. `view`s) are rejected up front rather
        # than reaching Mooncake, where reverse-mode silently returns a
        # `Mooncake.Tangent` and forward/Jacobian paths crash.
        problem = x -> sum(abs2, x)
        v = view([1.0, 2.0, 3.0], :)
        @test_throws r"dense vector" prepare(AutoMooncake(), problem, v)
        @test_throws r"dense vector" prepare(AutoMooncakeForward(), problem, v)
    end
end
