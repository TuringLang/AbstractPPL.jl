using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL:
    AbstractPPL,
    prepare,
    generate_testcases,
    generate_namedtuple_testcases,
    run_testcase,
    value_and_gradient!!
using ADTypes: AutoMooncake, AutoMooncakeForward
using Mooncake
using Test

# Known-broken paths in Mooncake:
#   * `value_and_jacobian!!` allocates fresh cotangent/Jacobian buffers on
#     every call (both modes); forward-mode Jacobian return type infers as
#     `Tuple{Any, Union{Array{T,3}, Matrix}}`.
#   * `value_and_gradient!!` on a context-lowered prep splats `args_to_zero`
#     per call (reverse mode allocates; forward mode also fails inference).
# Julia 1.10 also heap-allocates `Fix2`/closure captures that 1.11+ elides.
function _mooncake_alloc(case, adtype)
    if case.tag === :vector && case.jacobian !== nothing
        return :broken
    elseif case.tag === :context && adtype isa AutoMooncakeForward
        return :broken
    elseif VERSION < v"1.11"
        return :broken
    else
        return :test
    end
end
# The forward-mode Jacobian inference issue only affects non-empty input;
# the empty-input shortcut bypasses Mooncake and is inferable on either mode.
function _mooncake_inferred(case, adtype)
    is_jac_inf_broken =
        case.tag === :vector &&
        case.jacobian !== nothing &&
        length(case.x) > 0 &&
        adtype isa AutoMooncakeForward
    is_ctx_inf_broken = case.tag === :context && adtype isa AutoMooncakeForward
    return (is_jac_inf_broken || is_ctx_inf_broken) ? :broken : :test
end

@testset "AbstractPPLMooncakeExt" begin
    for (label, adtype) in (
        ("Mooncake (reverse)", AutoMooncake()),
        ("Mooncake (forward)", AutoMooncakeForward()),
    )
        @testset "$label" begin
            for case in generate_testcases()
                run_testcase(
                    case;
                    adtype,
                    atol=1e-6,
                    rtol=1e-6,
                    allocations=_mooncake_alloc(case, adtype),
                    type_stability=_mooncake_inferred(case, adtype),
                )
            end
            for case in generate_namedtuple_testcases()
                run_testcase(case; adtype, atol=1e-6, rtol=1e-6)
            end
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
