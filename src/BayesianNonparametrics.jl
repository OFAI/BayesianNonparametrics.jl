module BayesianNonparametrics

  using Distributions, Combinatorics, Clustering, ProgressMeter

  include("common.jl")
  include("math.jl")
  include("utils.jl")
  include("datasets.jl")
  include("distributions.jl")
  include("distfunctions.jl")
  include("inits.jl")
  include("inference.jl")
  include("models.jl")
  include("dpmm.jl")
  include("vipointestimate.jl")

end # module
