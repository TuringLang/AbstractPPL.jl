using Distributions # just added for testing

# # Adding interface for DAG representation using parts of SimplePPL/Mamba.jl
mutable struct Model
    value
    input
    eval
    kind
end

function Model(nt::T) where T <: NamedTuple
    ks = keys(nt)
    vals = [nt[k][i] for k in ks, i in 1:4] # create matrix of tuple values
    m = [(; zip(ks, vals[:,i])...) for i in 1:4] # zip each colum of vals with keys
    Model(m[1], m[2], m[3], m[4])
end

test = (
    a = ([0,1], (), () -> Normal(), :Stochastic), # should this explictly call rand(Normal())?
    b = (0, (), () -> 42, :Logical), 
    c = (0, (:a, :b), (a, b) -> MvNormal(a, 2sqrt(b)), :Stochastic)
)

m = Model(test)

m.value == (a = [0, 1], b = 0, c = 0)
m.input == (a = (), b = (), c = (:a, :b))
m.eval == (a = () -> D(), b = () -> 42, c = (a, b) -> Normal(a, 2sqrt(b)))
m.kind == (a = :Stochastic, b = :Logical, c = :Stochastic)

#m[@varname(a)] == (value = [0, 1], input = (), eval = ..., kind = Stochastic)
# @varname(a)::VarName{:a, IdentityLens}