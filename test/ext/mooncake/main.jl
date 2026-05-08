using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL: run_testcases
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
end
