export WishartGaussian, GaussianDiagonal, DirichletMultinomial, GammaNormal, NormalNormal, BetaBernoulli, ConjugatePostDistribution

abstract type ConjugatePostDistribution end

abstract type UnivariateConjugatePostDistribution <: ConjugatePostDistribution end
abstract type DiscreteUnivariateConjugatePostDistribution <: UnivariateConjugatePostDistribution end
abstract type ContinuousUnivariateConjugatePostDistribution <: UnivariateConjugatePostDistribution end

abstract type MultivariateConjugatePostDistribution <: ConjugatePostDistribution end
abstract type DiscreteMultivariateConjugatePostDistribution <: MultivariateConjugatePostDistribution end
abstract type ContinuousMultivariateConjugatePostDistribution <: MultivariateConjugatePostDistribution end

# Gaussian with Normal Inverse Wishart Prior
mutable struct WishartGaussian <: ContinuousMultivariateConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    sums::Vector{Float64}
    ssums::Array{Float64}

    # base model parameters
    μ0::Vector{Float64}
    κ0::Float64
    ν0::Float64
    Σ0::Array{Float64}

    function WishartGaussian(μ0::Vector{Float64}, κ0::Float64,
            ν0::Float64, Σ0::Array{Float64})

        d = length(μ0)
        new(d, 0, zeros(d), zeros(d, d), μ0, κ0, ν0, Σ0)
    end

end

# Normal with Gamma prior
mutable struct GammaNormal <: ContinuousUnivariateConjugatePostDistribution

	# sufficient statistics
	n::Int
	sums::Float64
    ssums::Float64

	# model parameters
	μ0::Float64
	λ0::Float64
	α0::Float64
	β0::Float64

	function GammaNormal(;μ0 = 0.0, λ0 = 1.0, α0 = 1.0, β0 = 1.0)
		new(0, 0.0, 0.0, μ0, λ0, α0, β0)
	end

end

# Normal with Normal prior
mutable struct NormalNormal <: ContinuousUnivariateConjugatePostDistribution

	# sufficient statistics
	n::Int
	sums::Float64
    ssums::Float64

	# model parameters
	μ0::Float64
	σ0::Float64

	function NormalNormal(;μ0 = 0.0, σ0 = 1.0)
		new(0, 0.0, 0.0, μ0, σ0)
	end

end

# Gaussian with Diagonal Covariance
mutable struct GaussianDiagonal{T <: ContinuousUnivariateConjugatePostDistribution} <: ContinuousMultivariateConjugatePostDistribution

    # sufficient statistics
    dists::Vector{T}

    # isn't the default constructor sufficient here?
    #function GaussianDiagonal(dists::Vector{T})
    #    new(dists)
    #end

end

# Multinomial with Dirichlet Prior
mutable struct DirichletMultinomial <: DiscreteMultivariateConjugatePostDistribution

    D::Int

    # sufficient statistics
    n::Int
    counts::SparseMatrixCSC{Int,Int}

    # base model parameters
    α0::Float64

  	# cache
  	dirty::Bool
  	Z2::Float64
  	Z3::Array{Float64}

    function DirichletMultinomial(D::Int, α0::Float64)
        new(D, 0, sparsevec(zeros(D)), α0, true, 0.0, Array{Float64}(0))
    end

end

# Bernoulli with Beta Prior
mutable struct BetaBernoulli <: DiscreteUnivariateConjugatePostDistribution

	# sufficient statistics
	successes::Int
    n::Int

	# beta distribution parameters
	α0::Float64
	β0::Float64

	function BetaBernoulli(;α0 = 1.0, β0 = 1.0)
		new(0, 0, α0, β0)
	end

end
