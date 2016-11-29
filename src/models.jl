export DPM, VCM, HDP

# define models

type DPM <: ModelType
  H::ConjugatePostDistribution
  α::Float64

  DPM(H::ConjugatePostDistribution; α = 1.0) = new(H, α)
end

type VCM <: ModelType
  α::Float64

  VCM(;α = 1.0) = new(α)
end

type HDP <: ModelType
  H::ConjugatePostDistribution
  α::Float64
  γ::Float64

  HDP(H::ConjugatePostDistribution; α = 0.1, γ = 5.0) = new(H, α, γ)
end
