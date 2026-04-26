using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using FiniteDifferences
using LogDensityProblems: LogDensityProblems
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

@testset "AbstractPPLFiniteDifferencesExt" begin
    fdm = FiniteDifferences.central_fdm(5, 1)
    adtype = ADTypes.AutoFiniteDifferences(; fdm)
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]
    values0 = (x=0.0, y=[0.0, 0.0])
    values = (x=3.0, y=[1.0, 2.0])

    run_shared_gradient_tests(
        adtype, x0, x; atol=1e-5, rtol=1e-5, test_autograd_kwargs=(; fdm)
    )
    run_shared_jacobian_tests(
        adtype, x0, [2.0, 3.0, 4.0]; atol=1e-5, rtol=1e-5, test_autograd_kwargs=(; fdm)
    )
    run_shared_namedtuple_tests(
        adtype, values0, values; atol=1e-5, rtol=1e-5, test_autograd_kwargs=(; fdm)
    )
    run_shared_invalid_mode_tests(adtype, x0)
    run_shared_ldp_tests(adtype, x0, x)
end
