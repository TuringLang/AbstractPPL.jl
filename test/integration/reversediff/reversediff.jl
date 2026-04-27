using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using ReverseDiff
using Test

include(joinpath(@__DIR__, "..", "..", "ext", "ad_tests.jl"))

struct StatefulQuadraticProblem
    data::Vector{Float64}
end

struct StatefulQuadraticPrepared
    data::Vector{Float64}
end

function AbstractPPL.prepare(problem::StatefulQuadraticProblem, x::AbstractVector{<:Real})
    return StatefulQuadraticPrepared(problem.data)
end

(p::StatefulQuadraticPrepared)(x::AbstractVector{<:Real}) = sum(abs2, x .- p.data)

function repeated_call_allocs(f)
    GC.gc()
    before = Base.gc_num()
    for _ in 1:100
        f()
    end
    after = Base.gc_num()
    return Base.GC_Diff(after, before).allocd
end

@testset "ReverseDiff via DifferentiationInterface" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(ADTypes.AutoReverseDiff(), x0, x)

    @testset "compiled prep reduces repeated-call allocations" begin
        problem = StatefulQuadraticProblem(randn(10))
        x0 = randn(10)
        x = randn(10)

        prepared_uncompiled = AbstractPPL.prepare(ADTypes.AutoReverseDiff(), problem, x0)
        prepared_compiled = AbstractPPL.prepare(
            ADTypes.AutoReverseDiff(; compile=true), problem, x0
        )

        @test prepared_uncompiled(x) ≈ prepared_compiled(x)

        val_uncompiled, grad_uncompiled = AbstractPPL.value_and_gradient(
            prepared_uncompiled, x
        )
        val_compiled, grad_compiled = AbstractPPL.value_and_gradient(prepared_compiled, x)
        @test val_compiled ≈ val_uncompiled
        @test grad_compiled ≈ grad_uncompiled

        allocs_uncompiled = repeated_call_allocs(
            () -> AbstractPPL.value_and_gradient(prepared_uncompiled, x)
        )
        allocs_compiled = repeated_call_allocs(
            () -> AbstractPPL.value_and_gradient(prepared_compiled, x)
        )

        @test allocs_compiled < allocs_uncompiled
    end
end
