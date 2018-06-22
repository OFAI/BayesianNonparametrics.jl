export PrecomputedInitialisation, RandomInitialisation, KMeansInitialisation, IncrementalInitialisation

mutable struct PrecomputedInitialisation <: InitialisationType
  Z::Array{Int}
  PrecomputedInitialisation(Z::Array{Int}) = new(Z)
end

mutable struct RandomInitialisation <: InitialisationType
  k::Int
  RandomInitialisation(;k = 2) = new(k)
end

mutable struct KMeansInitialisation <: InitialisationType
  k::Int
  maxiterations::Int
  KMeansInitialisation(;k = 2, maxiterations = 1000) = new(k, maxiterations)
end

mutable struct IncrementalInitialisation <: InitialisationType end
