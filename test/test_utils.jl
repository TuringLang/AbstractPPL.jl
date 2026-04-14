"""
Shared test utilities for AD extension tests.

Include this file inside `@testset` blocks in `test/ext/*/` tests after loading
AbstractPPL and Test.
"""

function _fd_gradient(f, x::AbstractVector)
    T = float(eltype(x))
    h = cbrt(eps(T))
    grad = Vector{T}(undef, length(x))
    for i in eachindex(x)
        xp = copy(x)
        xp[i] += h
        xm = copy(x)
        xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2h)
    end
    return grad
end

"""
    test_autograd(prepared, values; atol=1e-5, rtol=1e-5)

Compare `value_and_gradient(prepared, values)` against a central finite-difference
reference computed via the vector adapter `prepared(x::AbstractVector)`.
Calls `@test` internally; use inside a `@testset` block.
"""
function test_autograd(prepared, values::NamedTuple; atol=1e-5, rtol=1e-5)
    values = deepcopy(values)
    x = AbstractPPL.flatten_to_vec(values)
    f = x′ -> prepared(AbstractPPL.unflatten_from_vec(values, x′))
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, values)
    grad_fd_nt = AbstractPPL.unflatten_from_vec(values, _fd_gradient(f, x))
    @test val_ad ≈ f(x)
    for k in keys(values)
        @test getfield(grad_ad, k) ≈ getfield(grad_fd_nt, k) atol = atol rtol = rtol
    end
end
