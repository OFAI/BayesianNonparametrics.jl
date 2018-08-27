__precompile__(true)

module BayesianNonparametrics

  using Distributions#, Combinatorics, 
  using Clustering, ProgressMeter, StatsFuns
  
  # include Base modules
  using SparseArrays, Distributed, LinearAlgebra, Random

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
  include("hdp.jl")
  include("vipointestimate.jl")

end # module
