module AbstractPPLDifferentiationInterfaceLogDensityProblemsExt

using AbstractPPL: AbstractPPL
using AbstractPPL.Evaluators: Prepared, VectorEvaluator
using ADTypes: AbstractADType
using DifferentiationInterface: DifferentiationInterface as DI
using LogDensityProblems: LogDensityProblems

const _DIExt = Base.get_extension(AbstractPPL, :AbstractPPLDifferentiationInterfaceExt)

# Scalar-output `Prepared` from `prepare(adtype, scalar_f, x)`: DI populated
# `gradient_prep` and left `jacobian_prep === nothing`. `value_and_gradient!!`
# is structurally guaranteed to succeed for this shape, so advertise
# `LogDensityOrder{1}`. Vector-output preps and any other cache shape fall
# through to the base extension's `LogDensityOrder{0}` default.
function LogDensityProblems.capabilities(
    ::Type{<:Prepared{AD,E,C}}
) where {AD<:AbstractADType,E<:VectorEvaluator,F,GP,C<:_DIExt.DICache{F,GP,Nothing}}
    return LogDensityProblems.LogDensityOrder{1}()
end

end # module
