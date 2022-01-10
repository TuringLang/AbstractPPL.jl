
struct testData
    value::NamedTuple
    input::NamedTuple
    eval::NamedTuple
    kind::NamedTuple
end

struct testData2{T <: NamedTuple}
    value::T
    input::T
    eval::T
    kind::T
end

struct testData3{A, B, C, D <: NamedTuple}
    value::A
    input::B
    eval::C
    kind::D
end



ks = keys(test)
vals = [test[k][i] for k in ks, i in 1:4] # create matrix of tuple values
m = [(; zip(ks, vals[:,i])...) for i in 1:4]
for i in 1:4
    println(typeof(m[i]))
end

t = testData(m...)
t2 = testData2(m...)
t3 = testData3(m...)

@code_warntype t3.value

struct TestType{S, T <: NamedTuple}
    valI::S
    valF::T
end

v = (a = 1, b = 2, c = 3)
v2 = (a = 1.0, b = 2.0, c = 3.0)

t = TestType(v, v2)

@code_warntype t.valI

function gf(tt::TestType)
    return tt.valI
end

@code_warntype gf(t)