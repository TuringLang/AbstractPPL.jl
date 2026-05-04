module TestResources

import ..AbstractPPL: prepare

export TestCase, generate_testcases

struct QuadraticProblem end
struct QuadraticPrepared end
prepare(::QuadraticProblem, ::AbstractVector{<:Real}) = QuadraticPrepared()
(::QuadraticPrepared)(x::AbstractVector{<:Real}) = sum(xi -> xi^2, x)

struct VectorValuedProblem end
struct VectorValuedPrepared end
function prepare(::VectorValuedProblem, ::AbstractVector{<:Real})
    return VectorValuedPrepared()
end
(::VectorValuedPrepared)(x::AbstractVector{<:Real}) = [x[1] * x[2], x[2] + x[3]]

struct TestCase{F,XP,X,V,G,J,O,E}
    name::String
    f::F
    x_proto::XP
    x::X
    value::V
    gradient::G
    jacobian::J
    operation::O
    exception::E
end

function TestCase(name, f, x_proto, x, value, gradient, jacobian)
    return TestCase(name, f, x_proto, x, value, gradient, jacobian, :value, nothing)
end

function generate_testcases(::Val{:vector})
    return (
        TestCase(
            "quadratic (scalar output)",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0],
            14.0,
            [6.0, 2.0, 4.0],
            nothing,
        ),
        TestCase(
            "vector-valued (vector output)",
            VectorValuedProblem(),
            zeros(3),
            [2.0, 3.0, 4.0],
            [6.0, 7.0],
            nothing,
            [3.0 2.0 0.0; 0.0 1.0 1.0],
        ),
        TestCase(
            "empty input, scalar output",
            x -> 7.5,
            Float64[],
            Float64[],
            7.5,
            Float64[],
            nothing,
        ),
        TestCase(
            "empty input, vector output",
            x -> [2.0, 3.0],
            Float64[],
            Float64[],
            [2.0, 3.0],
            nothing,
            zeros(2, 0),
        ),
    )
end

function generate_testcases(::Val{:namedtuple})
    return (
        TestCase(
            "scalar output over (x::Real, y::Vector)",
            vs -> vs.x^2 + sum(abs2, vs.y),
            (x=0.0, y=zeros(2)),
            (x=3.0, y=[1.0, 2.0]),
            14.0,
            (x=6.0, y=[2.0, 4.0]),
            nothing,
        ),
        TestCase(
            "wrong NamedTuple structure",
            vs -> vs.x^2 + sum(abs2, vs.y),
            (x=0.0, y=zeros(2)),
            (x=3.0, z=[1.0, 2.0]),
            nothing,
            nothing,
            nothing,
            :gradient,
            r"same NamedTuple structure",
        ),
    )
end

function generate_testcases(::Val{:edge})
    return (
        TestCase(
            "wrong vector length",
            QuadraticProblem(),
            zeros(3),
            [3.0, 1.0, 2.0, 99.0],
            nothing,
            nothing,
            nothing,
            :call,
            DimensionMismatch,
        ),
        TestCase(
            "non-floating-point vector",
            QuadraticProblem(),
            zeros(3),
            [3, 1, 2],
            nothing,
            nothing,
            nothing,
            :call,
            r"floating-point",
        ),
        TestCase(
            "gradient of vector-valued output",
            VectorValuedProblem(),
            zeros(3),
            [2.0, 3.0, 4.0],
            nothing,
            nothing,
            nothing,
            :gradient,
            Exception,
        ),
    )
end

end # module
