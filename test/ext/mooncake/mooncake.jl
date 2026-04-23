using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using FiniteDifferences
using Mooncake
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

@testset "AbstractPPLMooncakeExt" begin
    for adtype in (
        ADTypes.AutoMooncake(; config=Mooncake.Config()),
        ADTypes.AutoMooncakeForward(; config=Mooncake.Config()),
    )
        @testset "$(nameof(typeof(adtype)))" begin
            x0 = zeros(3)
            x = [3.0, 1.0, 2.0]
            values0 = (x=0.0, y=[0.0, 0.0])
            values = (x=3.0, y=[1.0, 2.0])

            run_shared_gradient_tests(adtype, x0, x)
            run_shared_jacobian_tests(adtype, x0, [2.0, 3.0, 4.0])
            run_shared_namedtuple_tests(adtype, values0, values)
            run_shared_invalid_mode_tests(adtype, x0)

            @testset "Mooncake cache spec enforcement" begin
                prepared = AbstractPPL.prepare(adtype, QuadraticProblem(), x0)
                @test_throws Mooncake.PreparedCacheSpecError AbstractPPL.value_and_gradient(
                    prepared, [3.0, 1.0, 2.0, 3.0]
                )
            end
        end
    end
end
