export DPMHyperparam, DPMData

"Dirichlet Process Mixture Model Hyperparameters"
struct DPMHyperparam <: AbstractHyperparam

  γ_a::Float64
  γ_b::Float64

  "default values"
  DPMHyperparam(;α = 1.0, β = 1.0) = new(α, β)

end

"Dirichlet Process Mixture Model Data Object"
mutable struct DPMData <: AbstractModelData

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

mutable struct DPMBuffer <: AbstractModelBuffer

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

  # weights
  π::Vector{Float64}

  # number of active cluster
  K::Int

  # distributions
  G::Array{ConjugatePostDistribution}

  # base distribution
  G0::ConjugatePostDistribution

  # concentration parameter
  alpha::Float64
end

function init(X::AbstractArray{T}, model::DPM, init::KMeansInitialisation) where T <: Real

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

  K = init.k
  for k in sort(find(C .== 0), rev = true)
    deleteat!(G, k)
    Z[Z .> k] -= 1
    deleteat!(C, k)
    K -= 1
  end

  return DPMBuffer(
    collect(1:N),
    D,
    N,
    X,
    Z,
    C,
    rand(Dirichlet(C)),
    K,
    G,
    model.H,
    model.α)
end

function init(X::AbstractArray{T}, model::DPM, init::RandomInitialisation) where T <: Real

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
    rand(Dirichlet(C)),
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
    p = exp.(p - maximum(p))

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

  B.π = Float64[B.C[i] / (B.N + B.alpha - 1) for i in 1:B.K]

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
  model = DPMData(0.0, deepcopy(B.alpha), deepcopy(B.G), deepcopy(B.Z), copy(B.π))
  updateenergy!(model, B.X)
  model
end

function slicemap(i::Int, uu::Float64, ii::Int, π::Vector{Float64}, Zi::Int, Xi::AbstractArray,
    G::Vector{ConjugatePostDistribution}, G0::ConjugatePostDistribution, K::Int)
  if i == ii
		ui = uu
	else
		ui = rand(Uniform(uu, π[Zi]))
	end

	di = find(π .>= ui)
	p = [logpostpred(G[d], Xi)[1] for d in filter(d -> d <= K, di)]
	if any(di .> K)
		push!(p, logpostpred(G0, Xi)[1])
	end
	z = di[randomindex(exp(p - maximum(p)))]
	(z, i)
end

function slicesampling!(B::DPMBuffer)
	# sample global parameters
	n = counts(B.Z, 1:B.K)
	π = rand(Dirichlet(vcat(n, B.alpha)))
	b = [rand(Beta(1, n[k])) for k in 1:B.K]
	u = π[1:end-1].*b
	(uu, kk) = findmin(u)
	ii = rand(find(B.Z .== kk))

	# collect slices
	suffstat = @parallel (vcat) for i in 1:B.N
		slicemap(i, uu, ii, π, B.Z[i], view(B.X, i, :), B.G, B.G0, B.K)
	end

	# update GMM
	ki = 1
	emptyIds = Int[]

	for k in 1:B.K
		stats = filter(sstat -> sstat[1] == k, suffstat)
		if !isempty(stats)
			ids = reduce(vcat, (sstat[2] for sstat in stats))
			B.Z[ids] = ki
			B.G[ki] = add(B.G0, B.X[ids,:])
			ki += 1
		else
			push!(emptyIds, k)
		end
	end

	# remove empty clusters
	for eid in reverse(emptyIds)
		deleteat!(B.G, eid)
	end

	B.K = B.K - length(emptyIds)

	# add new components
	stats = filter(sstat -> sstat[1] > B.K + length(emptyIds), suffstat)

	if !isempty(stats)
		ids = reduce(vcat, (sstat[2] for sstat in stats))
		push!(B.G, add(B.G0, B.X[ids, :]))
		B.Z[ids] = B.K + 1
		B.K += 1
	end

  # update weights
  n = counts(B.Z, 1:B.K)
	B.π = rand(Dirichlet(n))
end
