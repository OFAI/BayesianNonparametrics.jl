export DPMHyperparam, DPMData

"Dirichlet Process Mixture Model Hyperparameters"
immutable DPMHyperparam <: AbstractHyperparam

  γ_a::Float64
  γ_b::Float64

  "default values"
  DPMHyperparam(;α = 1.0, β = 1.0) = new(α, β)

end

"Dirichlet Process Mixture Model Data Object"
type DPMData <: AbstractModelData

  # Energy
  energy::Float64

  # Dirichlet concentration parameter
  α::Float64

  # Distributions
  distributions::Array{ConjugatePostDistribution}

  # Assignments
  assignments::Array{Int}

  # Weights
  weights::Array{Float64}

end

type DPMBuffer <: AbstractModelBuffer

  # ids used for random access
  ids::Array{Int}

  # dimensionality of data
  D::Int

  # number of samples
  N::Int

  # samples
  X::AbstractArray

  # assignments
  Z::Array{Int}

  # number of samples per cluster
  C::Array{Int}

  # number of active cluster
  K::Int

  # distributions
  G::Array{ConjugatePostDistribution}

  # base distribution
  G0::ConjugatePostDistribution

  # concentration parameter
  alpha::Float64
end

function init{T <: Real}(X::AbstractMatrix{T}, model::DPM, init::KMeansInitialisation)

  (N, D) = size(X)

	if issparse(X)
    if !(T == Float64)
		  R = Clustering.kmeans(float(full(X')), init.k; maxiter = init.maxiterations)
    else
      R = Clustering.kmeans(full(X'), init.k; maxiter = init.maxiterations)
    end
	else
		R = Clustering.kmeans(X', init.k; maxiter = init.maxiterations)
	end

  Z = assignments(R)
  G = Array{ConjugatePostDistribution}(init.k)

  for c in 1:init.k
    idx = find(Z .== c)

    if length(idx) > 0
      G[c] = add(model.H, X[idx,:])
    else
      G[c] = deepcopy(model.H)
    end
  end

  C = zeros(Int, init.k)
  for i = 1:init.k
    C[i] = sum(Z .== i)
  end

  return DPMBuffer(
    collect(1:N),
    D,
    N,
    X,
    Z,
    C,
    init.k,
    G,
    model.H,
    model.α)
end

function init{T <: Real}(X::AbstractMatrix{T}, model::DPM, init::RandomInitialisation)

  (N, D) = size(X)

  Z = rand(1:init.k, N)

  G = Array{ConjugatePostDistribution}(init.k)

  for c in 1:init.k
    idx = find(Z .== c)

    if length(idx) > 0
      G[c] = add(model.H, X[idx,:])
    else
      G[c] = deepcopy(model.H)
    end
  end

  C = zeros(Int, init.k)
  for i = 1:init.k
    C[i] = sum(Z .== i)
  end

  return DPMBuffer(
    collect(1:N),
    D,
    N,
    X,
    Z,
    C,
    init.k,
    G,
    model.H,
    model.α)
end

"Single iteration of collabsed Gibbs sampling using CRP."
function gibbs!(B::DPMBuffer)

  # randomize data
  shuffle!(B.ids)

  z = -1
  k = -1

  for index in B.ids

    x = @view B.X[index, :]

    # get assignment
    z = B.Z[index]

    # remove sample from cluster
    remove!(B.G[z], x)

    # remove cluster assignment
    B.C[z] -= 1

    # udpate number of active clusters if necessary
    if B.C[z] < 1
      deleteat!(B.G, z)
      B.Z[B.Z .> z] -= 1
      deleteat!(B.C, z)
      B.K -= 1
    end

    # remove cluster assignment
    z = -1

    # compute posterior predictive
    # compute priors using chinese restaurant process
    # see: Samuel J. Gershman and David M. Blei, A tutorial on Bayesian nonparametric models.
    # In Journal of Mathematical Psychology (2012)
    p = ones(B.K + 1) * -Inf
    for i in 1:length(B.C)
      llh = logpostpred( B.G[i], x )[1]
      crp = log( B.C[i] / (B.N + B.alpha - 1) )

      p[i] = llh + crp
    end

    p[B.K + 1] = logpostpred(B.G0, x)[1] + log( B.alpha / (B.N + B.alpha - 1) )
    p = exp(p - maximum(p))

    k = randomindex(p)

    if k > length(B.G)
      # add new cluster
      Gk = add(B.G0, x)
      B.G = cat(1, B.G, Gk)
      B.C = cat(1, B.C, 0)
    else
      # add to cluster
      add!(B.G[k], x)
    end

    if k > B.K
      B.K += 1
    end

    B.C[k] += 1
    B.Z[index] = k
  end

  B
end

"Compute Energy of model for given data"
function updateenergy!(B::DPMData, X::AbstractArray)

  E = 0.00001

  for xi in 1:size(X, 1)

    pp = 0.0
    c = 0

    for i = 1:length(B.weights)
      p = exp( logpostpred( B.distributions[i], X[xi, :] )[1] ) * B.weights[i]

      # only sum over actual values (excluding nans)
      if p == p
        pp += p
        c += 1
      end
    end

    E += pp

  end

  B.energy = log( E / size(X, 1) )

end

function sampleparameters!(B::DPMBuffer, P::DPMHyperparam)
  B.alpha = resampleα(B.alpha, B.N, B.K, k = P.γ_a, Θ = P.γ_b, maxiter = 10)
end

function extractpointestimate(B::DPMBuffer)
  model = DPMData(0.0, deepcopy(B.alpha), deepcopy(B.G), deepcopy(B.Z), map(i -> B.C[i] / (B.N + B.alpha - 1), 1:B.K))
  updateenergy!(model, B.X)
  model
end
