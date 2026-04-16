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
    test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)

Compare `value_and_gradient(prepared, x)` against a central finite-difference
reference. Calls `@test` internally; use inside a `@testset` block.
"""
function test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, x)
    grad_fd = _fd_gradient(prepared, x)
    @test val_ad ≈ prepared(x)
    @test grad_ad ≈ grad_fd atol = atol rtol = rtol
end
