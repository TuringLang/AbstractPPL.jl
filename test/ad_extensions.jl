using AbstractPPL
using ADTypes: ADTypes
using Test

# ---------------------------------------------------------------------------
# Shared test problem: f(x, y) = x^2 + sum(y.^2)
# Gradient: (2x, 2y)
# ---------------------------------------------------------------------------
struct QuadraticProblem end

struct QuadraticPrepared end

function AbstractPPL.prepare(::QuadraticProblem, prototype::NamedTuple)
    return QuadraticPrepared()
end

# Type-stable for Enzyme: access fields directly instead of iterating values.
function (::QuadraticPrepared)(values::NamedTuple{(:x, :y)})
    return values.x^2 + sum(vi -> vi^2, values.y)
end

# ---------------------------------------------------------------------------
# flatten / unflatten unit tests
# ---------------------------------------------------------------------------
@testset "flatten / unflatten" begin
    nt = (x = 1.0, y = [2.0, 3.0])
    v = AbstractPPL.flatten_to_vec(nt)
    @test v == [1.0, 2.0, 3.0]
    nt2 = AbstractPPL.unflatten_from_vec(nt, v)
    @test nt2.x == 1.0
    @test nt2.y == [2.0, 3.0]

    # Nested NamedTuple
    nt3 = (a = 0.5, b = (c = 1.0, d = [2.0, 3.0]))
    v3 = AbstractPPL.flatten_to_vec(nt3)
    @test v3 == [0.5, 1.0, 2.0, 3.0]
    nt3r = AbstractPPL.unflatten_from_vec(nt3, v3)
    @test nt3r.a == 0.5
    @test nt3r.b.c == 1.0
    @test nt3r.b.d == [2.0, 3.0]

    # Matrix
    nt4 = (x = [1.0 2.0; 3.0 4.0],)
    v4 = AbstractPPL.flatten_to_vec(nt4)
    @test length(v4) == 4
    nt4r = AbstractPPL.unflatten_from_vec(nt4, v4)
    @test nt4r.x == [1.0 2.0; 3.0 4.0]
end

# ---------------------------------------------------------------------------
# Helper to run the standard gradient test suite on a prepared evaluator
# ---------------------------------------------------------------------------
function run_gradient_tests(prepared, backend_name)
    @testset "$backend_name" begin
        @test AbstractPPL.capabilities(prepared) == AbstractPPL.DerivativeOrder{1}()

        values = (x = 3.0, y = [1.0, 2.0])
        lp = prepared(values)
        @test lp ≈ 9.0 + 1.0 + 4.0  # 14.0

        val, grad = AbstractPPL.value_and_gradient(prepared, values)
        @test val ≈ 14.0
        @test grad.x ≈ 6.0     # 2*3
        @test grad.y ≈ [2.0, 4.0]  # 2*[1, 2]

        @test AbstractPPL.dimension(prepared) == 3

        # Vector adapter
        x_vec = [3.0, 1.0, 2.0]
        @test prepared(x_vec) ≈ 14.0
    end
end

# ---------------------------------------------------------------------------
# Backend-specific tests
# ---------------------------------------------------------------------------
@testset "AD extensions" begin
    problem = QuadraticProblem()
    prototype = (x = 0.0, y = [0.0, 0.0])

    @testset "ForwardDiff" begin
        using ForwardDiff
        adtype = ADTypes.AutoForwardDiff()
        prepared = AbstractPPL.prepare(adtype, problem, prototype)
        run_gradient_tests(prepared, "ForwardDiff")
    end

    @testset "Mooncake" begin
        using Mooncake
        adtype = ADTypes.AutoMooncake(; config=Mooncake.Config())
        prepared = AbstractPPL.prepare(adtype, problem, prototype)
        run_gradient_tests(prepared, "Mooncake")
    end

    @testset "Enzyme" begin
        using Enzyme
        adtype = ADTypes.AutoEnzyme()
        prepared = AbstractPPL.prepare(adtype, problem, prototype)
        run_gradient_tests(prepared, "Enzyme")
    end

    @testset "DifferentiationInterface fallback" begin
        using DifferentiationInterface
        # Verify DI path works directly (can't test extension dispatch since
        # native extensions take priority for all loaded backends).
        evaluator = AbstractPPL.prepare(problem, prototype)
        x0 = AbstractPPL.flatten_to_vec(prototype)
        f_vec = let evaluator = evaluator, prototype = prototype
            x -> evaluator(AbstractPPL.unflatten_from_vec(prototype, x))
        end
        di_backend = ADTypes.AutoForwardDiff()
        prep = DifferentiationInterface.prepare_gradient(f_vec, di_backend, x0)
        val, dx = DifferentiationInterface.value_and_gradient(f_vec, prep, di_backend, [3.0, 1.0, 2.0])
        @test val ≈ 14.0
        @test dx ≈ [6.0, 2.0, 4.0]
    end
end
