using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))
Pkg.instantiate()

using AbstractPPL
using ADTypes: ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Test

include(joinpath(@__DIR__, "..", "..", "autograd_tests.jl"))

# Stub backend without a native AbstractPPL extension; this exercises
# the AbstractPPLDifferentiationInterfaceExt catch-all dispatch.
struct DummyADType <: ADTypes.AbstractADType end
const adtype = DummyADType()

DI.prepare_gradient(f, ::DummyADType, x, ::DI.Constant) = Val(:gradient)
function DI.value_and_gradient(f, prep, ::DummyADType, x, ctx::DI.Constant)
    return (f(x, ctx.data), 2 .* x)
end
DI.prepare_jacobian(f, ::DummyADType, x, ::DI.Constant) = Val(:jacobian)
function DI.value_and_jacobian(f, prep, ::DummyADType, x, ctx::DI.Constant)
    jac = [
        x[2] x[1] zero(eltype(x))
        zero(eltype(x)) one(eltype(x)) one(eltype(x))
    ]
    return (f(x, ctx.data), jac)
end

@testset "AbstractPPLDifferentiationInterfaceExt" begin
    run_autograd_tests(adtype; atol=1e-6, rtol=1e-6)
end
