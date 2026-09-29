using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL:
    AbstractPPL, prepare, generate_testcases, run_testcase, value_and_gradient!!
using AbstractPPL.Evaluators: VectorEvaluator
using ADTypes: AutoForwardDiff
using ForwardDiff
using Test

# A problem whose own `prepare` attaches a cache, as a downstream package can.
struct AttachesCache end
_copy_then_square(x, buffer) = sum(abs2, copyto!(buffer, x))
function AbstractPPL.prepare(
    ::AttachesCache, x::AbstractVector{<:Real}; check_dims::Bool=true, context::Tuple=()
)
    return VectorEvaluator{check_dims}(_copy_then_square, length(x), context, (similar(x),))
end

@testset "AbstractPPLForwardDiffExt" begin
    @testset "ForwardDiff (default chunk)" begin
        for case in generate_testcases(Val(:vector))
            run_testcase(
                case;
                adtype=AutoForwardDiff(),
                atol=1e-6,
                rtol=1e-6,
                allocations=:test,
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

    # ForwardDiff rebuilds the target per call, so both the gradient and Hessian
    # entry points honour a call-time `context` override.
    @testset "call-time context override (#167)" begin
        for case in generate_testcases(Val(:context_override))
            run_testcase(case; adtype=AutoForwardDiff(), atol=1e-6, rtol=1e-6)
        end
    end

    @testset "cache is rejected" begin
        work = (; y=[2.0, 0.0], mu=zeros(2))
        @test_throws r"not supported by `AutoForwardDiff`" prepare(
            AutoForwardDiff(), (x, w) -> sum(abs2, x), [1.0, 2.0]; cache=(work,)
        )
        @test_throws r"not supported by `AutoForwardDiff`" prepare(
            AutoForwardDiff(), AttachesCache(), [1.0, 2.0]
        )
    end
end
