export add, add!, remove, remove!, posteriorParameters, logpostpred

"""
	add(d::ConjugatePostDistribution, X)

Add data to Posterior Distribution.
"""
function add(d::ConjugatePostDistribution, X)
	dist = deepcopy(d)
	add!(dist, X)
	return dist
end

"""
	add!(d::ConjugatePostDistribution, X)

Add data to Posterior Distribution (inplace).
"""
function add!(d::MultivariateConjugatePostDistribution, X::AbstractMatrix)
    @simd for i in 1:size(X, 1)
			@inbounds add!(d, @view X[i,:])
		end
end

function add!(d::UnivariateConjugatePostDistribution, X::AbstractMatrix)
	(N,D) = size(X)
	@assert D == 1
	add!(d, vec(X))
end

function add!(d::UnivariateConjugatePostDistribution, X::AbstractVector)
	@simd for i in 1:size(X, 1)
		@inbounds add!(d, X[i])
	end
end

function add!(d::WishartGaussian, X::AbstractVector)
	@assert length(X) == d.D

	d.n += 1
  d.sums += X
  d.ssums += X * X'
end

function add!(d::GaussianDiagonal, X::AbstractVector)
	for (dim, dist) in enumerate(d.dists)
		add!(dist, X[dim])
	end
end

function add!(d::ContinuousUnivariateConjugatePostDistribution, X::AbstractFloat)
	d.n += 1
	d.sums += X
	d.ssums += X^2
end

function add!(d::DirichletMultinomial, X)

    # process samples
    Dmin = minimum(X)
    Dmax = maximum(X)

    if Dmax > d.D
        throw(ArgumentError("Value of X and is larger than Multinomial Distribution!"))
    end

    if Dmin < 1
        throw(ArgumentError("Value of X and is smaller than 1!"))
    end

    d.n += length(X)

    for x in X
        d.counts[x] += 1
    end

    d.dirty = true
end

function add!(d::DirichletMultinomial, X::SparseVector{Int, Int})
		@assert length(X) == d.D

    d.n += 1
    d.counts += X
    d.dirty = true
end

function add!(d::BetaBernoulli, X::Integer)
	d.successes += X
	d.n += 1
end

"""
	remove(d::ConjugatePostDistribution, X::AbstractArray)

Remove data from Posterior Distribution.
"""
function remove(d::ConjugatePostDistribution, X)
		dist = deepcopy(d)
		remove!(dist, X)
		return dist
end

"""
	remove!(d::ConjugatePostDistribution, X)

Remove data from Posterior Distribution (inplace).
"""
function remove!(d::MultivariateConjugatePostDistribution, X::AbstractMatrix)
    @simd for i in 1:size(X, 1)
			@inbounds remove!(d, @view X[i,:])
		end
end

function remove!(d::UnivariateConjugatePostDistribution, X::AbstractMatrix)
	(N,D) = size(X)
	@assert D == 1
	remove!(d, vec(X))
end

function remove!(d::UnivariateConjugatePostDistribution, X::AbstractVector)
	@simd for i in 1:size(X, 1)
		@inbounds remove!(d, X[i])
	end
end

function remove!(d::WishartGaussian, X::AbstractVector)
		@assert length(X) == d.D

    d.n -= 1
    d.sums -= X
    d.ssums -= X * X'
end

function remove!(d::ContinuousUnivariateConjugatePostDistribution, X::AbstractFloat)
    d.n -= 1
    d.sums -= X
    d.ssums -= X^2
end

function remove!(d::GaussianDiagonal, X::AbstractVector)
	for (dim, dist) in enumerate(d.dists)
		remove!(dist, X[dim])
	end
end

function remove!(d::DirichletMultinomial, X)

    # process samples
    Dmin = minimum(X)
    Dmax = maximum(X)

    if Dmax > d.D
        throw(ArgumentError("Value of X and is larger than Multinomial Distribution!"))
    end

    if Dmin < 1
        throw(ArgumentError("Value of X and is smaller than 1!"))
    end

    d.n -= length(X)

    for x in X
        d.counts[x] -= 1
    end

    d.dirty = true
end

function remove!(d::DirichletMultinomial, X::SparseVector{Int, Int})

   # process samples
   @assert length(X) == d.D

   d.n -= 1
   d.counts -= X
   d.dirty = true
end

function remove!(d::BetaBernoulli, X::Bool)
		d.successes -= X
		d.n -= 1
end

"""
	isdistempty(d::ConjugatePostDistribution)

Check if distribution is empty (contains no samples).
"""
function isdistempty(d::ConjugatePostDistribution)
    return d.n <= 0
end

"""
	posteriorParameters(d::ConjugatePostDistribution)

Compute posterior distribution parameters.
"""
function posteriorParameters(d::WishartGaussian)
	if isdistempty(d)
	  return (d.μ0, d.κ0, d.ν0, d.Σ0)
	else

		# statistics
		sample_mu = d.sums / d.n

		# make sure values are not NaN
		sample_mu[sample_mu .!= sample_mu] = 0

		# compute posterior parameters
		κ = d.κ0 + d.n
		ν = d.ν0 + d.n
		μ = (d.κ0 * d.μ0 + d.n * sample_mu) / κ
		Σ = d.Σ0 + d.ssums - κ * (μ * μ') + ( d.κ0 * (d.μ0 * d.μ0') )

		return (μ, κ, ν, Σ)
	end
end

function posteriorParameters(d::GammaNormal)
	if isdistempty(d)
	  return (d.μ0, d.λ0, d.α0, d.β0)
	else

		# statistics
    sample_mu = d.sums / d.n

    # make sure values are not NaN
    sample_mu = sample_mu != sample_mu ? 0 : sample_mu

    # compute posterior parameters
    μ = (d.λ0 * d.μ0 + d.sums) / (d.λ0 + d.n)
    λ = d.λ0 + d.n
    α = d.α0 + (d.n / 2)
    s = (d.ssums / d.n) - (sample_mu * sample_mu)
    β = d.β0 + 1/2 * (d.n * s + (d.λ0 * d.n * (sample_mu - d.μ0)^2 ) / (d.λ0 + d.n) )

		return (μ, λ, α, β)
	end
end

function posteriorParameters(d::NormalNormal)
	if isdistempty(d)
	  return (d.μ0, d.σ0)
	else

		# statistics
	  sample_mu = d.sums / d.n
		sample_var = (d.n * d.ssums - d.sums^2) / (d.n*(d.n-1))
		sample_var = (d.ssums - (d.sums^2)/d.n) / (d.n - 1)
		sample_var = 10.0

		# make sure values are not NaN
		sample_mu = sample_mu != sample_mu ? 0 : sample_mu

		σ = (sample_var * d.σ0) / (d.n * d.σ0 + sample_var)
	  μ = σ * (d.n*sample_mu/sample_var + d.μ0/d.σ0)

		return (μ, σ)
	end
end

function posteriorParameters(d::BetaBernoulli)
	α = d.α + d.successes
	β = d.β + d.n - d.successes
	return (α, β)
end

"""
	logpostpred(d::ConjugatePostDistribution, X)

Compute log posterior predictive.
"""
function logpostpred(d::WishartGaussian, x)

	(μ, κ, ν, Σ) = posteriorParameters(d)

  # posterior predictive of Normal Inverse Wishart is student-t Distribution, see:
  # K. Murphy, Conjugate Bayesian analysis of the Gaussian distribution. Eq. 258
	C = Σ * ((κ + 1) / (κ * (ν - d.D + 1)))
	@assert isposdef(C)

  return Distributions.logpdf(Distributions.MvTDist(ν - d.D + 1, μ, C), x)
end

function logpostpred(d::GammaNormal, x)
	return Float64[logpostpred(d, xi) for xi in x]
end

function logpostpred(d::GammaNormal, x::Number)

   (μ, λ, α, β) = posteriorParameters(d)

    # posterior predictive of Normal Gamma is student-t Distribution
    df = 2 * α
    mean = μ
    sigma = ( β * (λ + 1) ) / (λ * α)

		return tlogpdf(x, df, mean, sigma)
end

function logpostpred(d::GaussianDiagonal, x)
	return vec(sum([logpostpred(di, x[dim, :]) for (dim, di) in enumerate(d.dists)]))
end

function logpostpred(d::NormalNormal, x)
	return Float64[logpostpred(d, xi) for xi in x]
end

function logpostpred(d::NormalNormal, x::Number)

	(μ, σ) = posteriorParameters(d)

	return normlogpdf(μ, sqrt(σ), x)
end

"Log PMF for MultinomialDirichlet."
function logpostpred(d::DirichletMultinomial, x)

    N = length(x)

	  # TODO: This is bad and slow code, improve!!!
    # construct sparse vector
    xx = spzeros(d.D, N)
    for i in 1:N
      xx[x[i]] += 1
    end

		m = sum(xx)
		mi = nnz(xx)

		if d.dirty
			d.Z2 = lgamma(d.α0 + d.n)
			d.Z3 = lgamma( d.α0/d.D + d.counts )
			d.dirty = false
		end

	 l1 = lgamma(m + 1) - sum(lgamma(mi + 1))
	 l2 = d.Z2 - lgamma(d.α0 + d.n + m)
	 l3 = sum( lgamma( d.α0 / d.D + d.counts + xx ) - d.Z3 )

	 return [l1 + l2 + l3]
end

"Log PMF for MultinomialDirichlet."
function logpostpred(d::DirichletMultinomial, x::SparseVector{Int, Int})

   D = size(x, 1)
   N = size(x, 2)

   if N > 1
      throw(ErrorException("Multiple samples are not supported yet!"))
   end

   m = sum(x)
   mi = nnz(x)

   if d.dirty
      d.Z2 = lgamma(d.α0 + d.n)
      d.Z3 = lgamma( d.α0/d.D + d.counts )

      d.dirty = false
   end

	l1 = lgamma(m + 1) - sum(lgamma(mi + 1))
	l2 = d.Z2 - lgamma(d.α0 + d.n + m)
	l3 = sum( lgamma( d.α0 / d.D + d.counts + x ) - d.Z3 )

   return [l1 + l2 + l3]
end

"Log PMF for BernoulliBeta."
function logpostpred(d::BetaBernoulli, X::AbstractArray)
	return Float64[logpostpred(d, x) for x in X]
end

function logpostpred(d::BetaBernoulli, X::Bool)

  	# posterior
		(α, β) = posteriorParameters(d)

   return log(α) - log(α + β)
end
