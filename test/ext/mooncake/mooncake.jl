using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using Mooncake
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

@testset "AbstractPPLMooncakeExt" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(ADTypes.AutoMooncake(), x0, x)
    run_shared_jacobian_tests(ADTypes.AutoMooncake(), x0, [2.0, 3.0, 4.0])
    run_shared_jacobian_tests(ADTypes.AutoMooncakeForward(), x0, [2.0, 3.0, 4.0])
end
