using AbstractPPL
using AbstractPPL: prepare, value_and_gradient!!, evaluate!!
using AbstractPPL.Evaluators: Prepared, VectorEvaluator, NamedTupleEvaluator
using ADTypes: ADTypes
using Test

struct DummyModel end

struct DummyPrepared
    prototype_keys::Tuple
end

function AbstractPPL.prepare(model::DummyModel, values::NamedTuple)
    return DummyPrepared(keys(values))
end

function (p::DummyPrepared)(values::NamedTuple)
    keys(values) == p.prototype_keys ||
        error("expected fields $(p.prototype_keys), got $(keys(values))")
    return sum(x -> x isa AbstractArray ? sum(x) : x, values)
end

struct DummyADType <: ADTypes.AbstractADType end

function AbstractPPL.prepare(
    adtype::DummyADType, model::DummyModel, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    f = x -> sum(x)
    return Prepared(adtype, VectorEvaluator{check_dims}(f, length(x)))
end

function AbstractPPL.value_and_gradient!!(
    p::Prepared{DummyADType}, x::AbstractVector{<:Real}
)
    return (sum(x), ones(length(x)))
end

@testset "Evaluators interface" begin
    @testset "explicit evaluator shapes" begin
        ve = AbstractPPL.Evaluators.VectorEvaluator(sum, 3)
        @test ve([1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch ve([1.0, 2.0])
        @test_throws r"floating-point" ve([1, 2, 3])

        ne = AbstractPPL.Evaluators.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test ne((a=1.0, b=[2.0, 3.0])) == 6.0
        @test ne.inputspec == (a=0.0, b=zeros(2))
        @test_throws MethodError ne([1.0, 2.0, 3.0])

        # `CheckInput=false` skips the per-call shape checks.
        ve_unchecked = AbstractPPL.Evaluators.VectorEvaluator{false}(sum, 3)
        @test ve_unchecked([1.0, 2.0]) == 3.0

        ne_unchecked = AbstractPPL.Evaluators.NamedTupleEvaluator{false}(
            x -> 0.0, (a=0.0, b=zeros(2))
        )
        @test ne_unchecked((totally=:wrong,)) == 0.0
        @test_throws r"same NamedTuple structure" ne((totally=:wrong,))

        # Nested array shape: same `typeof` (Vector{Float64}), different size.
        @test_throws r"Nested array" ne((a=1.0, b=[2.0]))

        # Array-of-arrays: same `typeof` and outer size, mismatched inner size.
        ne_nested = AbstractPPL.Evaluators.NamedTupleEvaluator(
            x -> sum(sum, x.b), (b=[zeros(2), zeros(2)],)
        )
        @test ne_nested((b=[[1.0, 2.0], [3.0, 4.0]],)) == 10.0
        @test_throws r"Nested array" ne_nested((b=[[1.0], [2.0]],))

        # Unsupported leaf types are rejected rather than silently passing.
        ne_string = AbstractPPL.Evaluators.NamedTupleEvaluator(x -> length(x.s), (s="abc",))
        @test_throws r"Supported leaves" ne_string((s="abcde",))

        # `_check_ad_input` is dispatch-gated by `CheckInput` so the AD hot
        # path pays nothing when the evaluator was prepared with
        # `check_dims=false`.
        ve_checked = AbstractPPL.Evaluators.VectorEvaluator{true}(sum, 3)
        @test AbstractPPL.Evaluators._check_ad_input(ve_checked, [1.0, 2.0, 3.0]) ===
            nothing
        @test_throws DimensionMismatch AbstractPPL.Evaluators._check_ad_input(
            ve_checked, [1.0, 2.0]
        )
        @test_throws r"floating-point" AbstractPPL.Evaluators._check_ad_input(
            ve_checked, [1, 2, 3]
        )
        @test AbstractPPL.Evaluators._check_ad_input(ve_unchecked, [1.0, 2.0]) === nothing
        @test AbstractPPL.Evaluators._check_ad_input(ve_unchecked, [1, 2, 3]) === nothing
    end

    @testset "prepare (structural)" begin
        model = DummyModel()
        values = (x=0.0, y=[1.0, 2.0])
        prepared = prepare(model, values)
        @test prepared isa DummyPrepared
        @test prepared.prototype_keys == (:x, :y)

        lp = prepared((x=0.5, y=[1.5, 2.5]))
        @test lp ≈ 0.5 + 1.5 + 2.5

        @test_throws ErrorException prepared((a=1.0, b=2.0))

        # Generic fallback wraps a plain callable in the appropriate evaluator
        # so per-call shape checks fire even without a backend-specific override.
        pv = prepare(sum, zeros(3))
        @test pv isa VectorEvaluator{true}
        @test pv([1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch pv([1.0, 2.0])

        ntfun = v -> v.a + sum(v.b)
        pn = prepare(ntfun, (a=0.0, b=zeros(2)))
        @test pn isa NamedTupleEvaluator{true}
        @test pn((a=1.0, b=[2.0, 3.0])) == 6.0

        # check_dims=false propagates to the wrapper.
        pv_unchecked = prepare(sum, zeros(3); check_dims=false)
        @test pv_unchecked isa VectorEvaluator{false}
        @test pv_unchecked([1.0, 2.0]) == 3.0  # wrong length, no error
    end

    @testset "prepare (AD-aware)" begin
        model = DummyModel()
        x0 = zeros(3)
        adtype = DummyADType()
        prepared = prepare(adtype, model, x0)
        @test prepared isa Prepared{DummyADType}

        x = [0.5, 1.5, 2.5]
        @test prepared(x) ≈ 0.5 + 1.5 + 2.5

        val, grad = value_and_gradient!!(prepared, x)
        @test val ≈ 0.5 + 1.5 + 2.5
        @test grad ≈ [1.0, 1.0, 1.0]

        # check_dims=false skips the per-call dimension check.
        prepared_unchecked = prepare(adtype, model, x0; check_dims=false)
        @test prepared_unchecked([1.0, 2.0]) ≈ 3.0  # wrong length, no error
    end

    @testset "missing AD package extensions" begin
        model = DummyModel()
        x0 = zeros(3)

        @test_throws MethodError AbstractPPL.Evaluators.prepare(
            ADTypes.AutoEnzyme(), model, x0
        )
    end

    @testset "evaluate!!" begin
        ve = AbstractPPL.Evaluators.VectorEvaluator(sum, 3)
        @test evaluate!!(ve, [1.0, 2.0, 3.0]) == 6.0
        @test_throws DimensionMismatch evaluate!!(ve, [1.0, 2.0])

        ne = AbstractPPL.Evaluators.NamedTupleEvaluator(
            x -> x.a + sum(x.b), (a=0.0, b=zeros(2))
        )
        @test evaluate!!(ne, (a=1.0, b=[2.0, 3.0])) == 6.0

        adtype = DummyADType()
        prepared = prepare(adtype, DummyModel(), zeros(3))
        @test evaluate!!(prepared, [0.5, 1.5, 2.5]) ≈ 4.5
    end
end
