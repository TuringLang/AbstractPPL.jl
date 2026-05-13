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
        end
    end

    @testset "raw_gradient_target" begin
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
            ad,
            problem,
            x;
            check_dims=false,
            raw_gradient_target=(raw_logdensity, (problem.offset,)),
        )

        # `prepared(x)` still calls `problem(x)` on both paths.
        @test generic(x) == problem(x)
        @test lowered(x) == problem(x)

        # Same value and gradient as the generic path.
        @test value_and_gradient!!(generic, x) == value_and_gradient!!(lowered, x)

        # Rejects on forward mode, vector-valued problems, and empty input.
        vec_problem = x -> [x[1]^2, x[1] + 1.0]
        @test_throws ArgumentError prepare(
            AutoMooncakeForward(; config=nothing),
            problem,
            x;
            check_dims=false,
            raw_gradient_target=(raw_logdensity, (problem.offset,)),
        )
        @test_throws ArgumentError prepare(
            ad,
            vec_problem,
            x;
            check_dims=false,
            raw_gradient_target=((y, c) -> [y[1] * c], (1.0,)),
        )
        @test_throws ArgumentError prepare(
            ad,
            problem,
            Float64[];
            check_dims=false,
            raw_gradient_target=(raw_logdensity, (problem.offset,)),
        )

        # Jacobian on a scalar-only lowered cache surfaces our arity-mismatch error.
        @test_throws r"vector-valued" AbstractPPL.value_and_jacobian!!(lowered, x)
    end
end
