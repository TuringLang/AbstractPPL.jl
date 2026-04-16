"""
Shared test utilities for AD extension tests.

Include this file inside `@testset` blocks in `test/ext/*/` tests after loading
AbstractPPL and Test.
"""

"""
    test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)

Compare `value_and_gradient(prepared, x)` against a finite-difference
reference. Calls `@test` internally; use inside a `@testset` block.
"""
function test_autograd(prepared, x::AbstractVector; atol=1e-5, rtol=1e-5)
    val_ad, grad_ad = AbstractPPL.value_and_gradient(prepared, x)
    grad_fd = AbstractPPL.test_grad(prepared, x)
    @test val_ad ≈ prepared(x)
    return isnothing(grad_fd) || @test grad_ad ≈ grad_fd atol = atol rtol = rtol
end
