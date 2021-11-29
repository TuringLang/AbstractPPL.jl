include("simpleppl.jl")

model = Model(

  y = Stochastic(1,
    (mu, s2) ->  MvNormal(mu, sqrt(s2)),
    false
  ),

  mu = Logical(1,
    (xmat, beta) -> xmat * beta,
    false
  ),

  # Functions should be defined to take, as their arguments, the inputs upon which their
  # nodes depend and, for stochastic nodes, return distribution objects or arrays of objects
  # compatible with the Distributions package.


  # Arg 1: Absence of an integer value implies a scalar node.

  # Arg 3: An optional boolean argument after the function can be specified to indicate whether
  beta = Stochastic(1,
  # values of the node should be monitored (saved) during MCMC simulations
    () -> MvNormal(2, sqrt(1000))
  ),

  s2 = Stochastic( # scalar node.
    () -> InverseGamma(2.0, 3.0)
  )

)