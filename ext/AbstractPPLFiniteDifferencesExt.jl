module AbstractPPLFiniteDifferencesExt

using AbstractPPL: AbstractPPL
using AbstractPPL.ADProblems:
    _assert_gradient_output, _assert_jacobian_output, _assert_namedtuple_shape
using AbstractPPL.Utils: flatten_to!!, unflatten_to!!
using ADTypes: AutoFiniteDifferences
using FiniteDifferences: FiniteDifferences

const DEFAULT_TEST_FDM = FiniteDifferences.central_fdm(5, 1)

function _test_autograd_ref(p, x::AbstractVector{<:Real}, fdm=DEFAULT_TEST_FDM)
    return (p(x), FiniteDifferences.grad(fdm, p, x)[1])
end

function _test_autograd_ref(p, values::NamedTuple, fdm=DEFAULT_TEST_FDM)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    f = x -> p.evaluator(unflatten_to!!(p.evaluator.inputspec, x))
    val = p.evaluator(values)
    grad = FiniteDifferences.grad(fdm, f, x)[1]
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

_flat_grad(g) = g
_flat_grad(g::NamedTuple) = flatten_to!!(nothing, g)

function _assert_test_autograd_matches(
    val_ad, grad_ad, val_fd, grad_fd; atol=1e-5, rtol=1e-5
)
    isapprox(val_ad, val_fd; atol=atol, rtol=rtol) || throw(
        ArgumentError(
            "Value mismatch against finite differences: got $val_ad, expected $val_fd."
        ),
    )
    isapprox(_flat_grad(grad_ad), _flat_grad(grad_fd); atol=atol, rtol=rtol) || throw(
        ArgumentError(
            "Gradient mismatch against finite differences with atol=$atol and rtol=$rtol.",
        ),
    )
    return nothing
end

function AbstractPPL.test_autograd(
    prepared, x::AbstractVector; atol=1e-5, rtol=1e-5, fdm=DEFAULT_TEST_FDM
)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, x)
    val_fd, grad_fd = _test_autograd_ref(prepared, x, fdm)
    return _assert_test_autograd_matches(val_ad, grad_ad, val_fd, grad_fd; atol, rtol)
end

function AbstractPPL.test_autograd(
    prepared, values::NamedTuple; atol=1e-5, rtol=1e-5, fdm=DEFAULT_TEST_FDM
)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, values)
    val_fd, grad_fd = _test_autograd_ref(prepared, values, fdm)
    return _assert_test_autograd_matches(val_ad, grad_ad, val_fd, grad_fd; atol, rtol)
end

struct FDPrepared{E,F,M} <: AbstractPPL.ADProblems.AbstractPrepared
    evaluator::E
    f::F
    fdm::M
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, values::NamedTuple; check_dims::Bool=true
)
    evaluator = AbstractPPL.ADProblems.NamedTupleEvaluator{check_dims}(
        AbstractPPL.prepare(problem, values), values
    )
    f = x -> evaluator(unflatten_to!!(values, x))
    return FDPrepared(evaluator, f, adtype.fdm)
end

function AbstractPPL.prepare(
    adtype::AutoFiniteDifferences, problem, x::AbstractVector{<:Real}; check_dims::Bool=true
)
    raw = AbstractPPL.prepare(problem, x)
    length(x) == 0 && return AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, 0)
    evaluator = AbstractPPL.ADProblems.VectorEvaluator{check_dims}(raw, length(x))
    return FDPrepared(evaluator, evaluator, adtype.fdm)
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.NamedTupleEvaluator}, values::NamedTuple
)
    _assert_namedtuple_shape(p.evaluator, values)
    x = flatten_to!!(nothing, values)
    val = p.evaluator(values)
    _assert_gradient_output(val)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, unflatten_to!!(p.evaluator.inputspec, grad))
end

function AbstractPPL.value_and_gradient(
    p::FDPrepared{<:AbstractPPL.ADProblems.VectorEvaluator}, x::AbstractVector{<:Real}
)
    val = p.evaluator(x)
    _assert_gradient_output(val)
    grad = FiniteDifferences.grad(p.fdm, p.f, x)[1]
    return (val, grad)
end

function AbstractPPL.value_and_jacobian(
    p::FDPrepared{<:AbstractPPL.ADProblems.VectorEvaluator}, x::AbstractVector{<:Real}
)
    val = p.evaluator(x)
    _assert_jacobian_output(val)
    jac = FiniteDifferences.jacobian(p.fdm, p.f, x)[1]
    return (val, jac)
end

end # module
