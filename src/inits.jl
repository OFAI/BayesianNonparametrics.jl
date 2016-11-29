export PrecomputedInitialisation, RandomInitialisation, KMeansInitialisation, IncrementalInitialisation

type PrecomputedInitialisation <: InitialisationType
  Z::Array{Int}
  PrecomputedInitialisation(Z::Array{Int}) = new(Z)
end

type RandomInitialisation <: InitialisationType
  k::Int
  RandomInitialisation(;k = 2) = new(k)
end

type KMeansInitialisation <: InitialisationType
  k::Int
  maxiterations::Int
  KMeansInitialisation(;k = 2, maxiterations = 1000) = new(k, maxiterations)
end

type IncrementalInitialisation <: InitialisationType end
