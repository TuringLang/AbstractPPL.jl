using AbstractPPL
import AbstractPPL.GraphPPL:GraphInfo, Model, get_dag, set_node_value!, 
                            get_node_value, get_sorted_vertices, get_node_eval,
                            get_nodekind, get_node_input, get_model_values, 
                            set_model_values, rand, rand!, logdensityof, step, 
                            get_namemap, flatten_samples, Chains
using AbstractMCMC
using AdvancedMH
using Distributions
using LinearAlgebra

f(a, x, b) = ( a .* x ) .+ b

data = f(2.0, collect(1.0:10.0), 5.0) .+ randn(10)

m = Model(
    a = (0., () -> Normal(0.0, 5.0), :Stochastic), 
    x = (collect(1.0:10.0), () -> collect(1.0:10.0), :Logical),
    b = (0., () -> Normal(0.0, 5.0), :Stochastic),
    y = (data, (a, x, b) -> MvNormal(f(a, x, b), 1.0), :Stochastic)
)

# add a separate nodekind for Observations that are not updated
# during random sampling

rand!(m)

spl = RWMH(MvNormal(zeros(2), I))
samples = sample(m, spl, 1_0)
plot(samples)