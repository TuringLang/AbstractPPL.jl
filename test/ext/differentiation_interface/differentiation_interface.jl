using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface
using FiniteDifferences
using LogDensityProblems: LogDensityProblems
using Test

include(joinpath(@__DIR__, "..", "ad_tests.jl"))

# Use a backend without a native AbstractPPL extension so this test exercises
# AbstractPPLDifferentiationInterfaceExt dispatch directly.
const fdm = FiniteDifferences.central_fdm(5, 1)
struct DummyADType <: ADTypes.AbstractADType end
const adtype = DummyADType()

function DifferentiationInterface.prepare_gradient(f, ::DummyADType, x)
    return DifferentiationInterface.prepare_gradient(
        f, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

function DifferentiationInterface.value_and_gradient(f, prep, ::DummyADType, x)
    return DifferentiationInterface.value_and_gradient(
        f, prep, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

function DifferentiationInterface.prepare_jacobian(f, ::DummyADType, x)
    return DifferentiationInterface.prepare_jacobian(
        f, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

function DifferentiationInterface.value_and_jacobian(
    f, prep::DifferentiationInterface.JacobianPrep, ::DummyADType, x
)
    return DifferentiationInterface.value_and_jacobian(
        f, prep, ADTypes.AutoFiniteDifferences(; fdm), x
    )
end

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    x0 = zeros(3)
    x = [3.0, 1.0, 2.0]

    run_shared_gradient_tests(
        adtype,
        x0,
        x;
        atol=1e-6,
        rtol=1e-6,
        test_autograd_kwargs=(; atol=1e-4, rtol=1e-4, fdm),
    )
    run_shared_jacobian_tests(adtype, x0, [2.0, 3.0, 4.0]; atol=1e-6, rtol=1e-6)
    run_shared_invalid_mode_tests(adtype, x0)
    run_shared_ldp_tests(adtype, x0, x)
end
