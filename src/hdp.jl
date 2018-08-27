export HDPHyperparam

"Hierarchical Dirichlet Process Mixture Model Hyperparameters"
struct HDPHyperparam <: AbstractHyperparam

  γ_a::Float64
  γ_b::Float64

  α_a::Float64
  α_b::Float64

  "default values"
  HDPHyperparam(;γ_a = 1.0, γ_b = 1.0, α_a = 1.0, α_b = 1.0) = new(γ_a, γ_b, α_a, α_b)

end

"Hierarchical Dirichlet Process Mixture Model Data Object"
mutable struct HDPData <: AbstractModelData

  # Energy
  energy::Float64

  # Distributions
  distributions::Vector{ConjugatePostDistribution}

  # Assignments
  assignments::Vector{Vector{Int}}

	# Weights
  weights::Vector{Vector{Float64}}

end

mutable struct HDPBuffer{T <: Real} <: AbstractModelBuffer

  # samples
  X::Vector{Vector{T}}

  # number of groups
  N0::Int

  # indecies of groups
  N0idx::Vector{Int}

  # number of samples per group
  Nj::Vector{Int}

  # indecies of samples per group
  Njidx::Array{Vector}

  # assignments
  Z::Vector{Vector{Int}}

  # number of active cluster
  K::Int

  # number of samples per group per cluster
  C::Array{Int, 2}

  # total number of tables
  totalnt::Array{Int}

  # number of clusters per table
  classnt::Array{Int}

  # distributions
  G::Vector{ConjugatePostDistribution}

  # base distribution
  G0::ConjugatePostDistribution

  # concentration parameter
  α::Float64

  # concentration parameters of clusters
  β::Array{Float64}

  # gamma
  γ::Float64
end


"HDP initialization using random assignments."
function init(X::Vector{Vector{T}}, model::HDP, init::RandomInitialisation) where T <: Real

    Z = Vector{Vector{Int}}(length(X))
    G = ConjugatePostDistribution[]

    K = init.k

    for i in 1:length(X)
      N = length(X[i])
      Z[i] = rand(1:K, N)

      for c in 1:K
        idx = find(Z[i] .== c)
        if length(idx) > 0
          if size(G, 1) < c
            push!(G, add(model.H, X[i][idx]))
          else
            add!(G[c], X[i][idx])
          end
        end
      end
    end

    N0 = length(X)
    N0idx = collect(1:N0)

    Nj = [length(x) for x in X]
    Njidx = [collect(1:N) for N in Nj]

    C = reduce(hcat, [counts(z, 1:K) for z in Z])

    # init step
    β = ones(K) ./ K
    classnt = randnumtable(model.α * β[:,ones(Int, N0)]', C')
    totalnt = sum(classnt, 1)

    # set alpha vector of Dirichlet Distribution to sample β
    a = zeros(K + 1)
    a[1:end-1] = totalnt
    a[end] = model.γ

    # update beta
    β = rand(Dirichlet(a));

    return HDPBuffer{T}(
      X,
      N0,
      N0idx,
      Nj,
      Njidx,
      Z,
      K,
      C,
      totalnt,
      classnt,
      G,
      model.H,
      model.α,
      β,
      model.γ)
end

"Single Gibbs sweep of HDP training using beta variables."
function gibbs!(B::HDPBuffer)

    shuffle!(B.N0idx)
    prob = zeros(B.K + 1) * -Inf

    for j in B.N0idx

        # get samples of group
        data = B.X[j]
        z = B.Z[j]

        shuffle!(B.Njidx[j])

        for i in B.Njidx[j]

            cluster = z[i]

            # remove data item from model
            remove!(B.G[cluster], data[i])
            B.C[cluster, j] -= 1

            if isdistempty(B.G[cluster])

                # remove cluster
                B.G = B.G[[1:cluster-1; cluster+1:end]]

                for jj in 1:B.N0
                    B.Z[jj][B.Z[jj] .> cluster] -= 1
                end

                B.β = B.β[[1:cluster-1; cluster+1:end]]

                B.classnt = B.classnt[:,[1:cluster-1; cluster+1:end]]
                B.totalnt = B.totalnt[[1:cluster-1; cluster+1:end]]'

                B.K -= 1

                B.C = B.C[[1:cluster-1; cluster+1:end],:]
                prob = zeros(B.K + 1) * -Inf
            end

            # compute log likelihood
            for k in 1:B.K
              llh = logpostpred(B.G[k], data[i])[1]
              prior = B.C[k, j] + B.β[k] * B.α
              prob[k] = llh + log( prior )
            end

            prob[B.K + 1] = logpostpred(B.G0, data[i])[1] + log( B.β[B.K+1] * B.α )
            prob = exp(prob - maximum(prob))

            c = randomindex(prob)

            # add data to model
            B.Z[j][i] = c

            if c > B.K
                # create new cluster
                B.K += 1
                B.G = cat(1, B.G, deepcopy(B.G0))
                b = rand(Dirichlet([1, B.γ]))
                b = b * B.β[end]
                B.β = cat(1, B.β, 1)
                B.β[end-1:end] = b
                B.C = cat(1, B.C, zeros(Int, 1, B.N0))
                prob = zeros(B.K + 1) * -Inf
            end

            B.C[c, j] += 1
            add!(B.G[c], [data[i]])

        end

    end

    # sample number of tables
    kk = maximum([0, B.K - length(B.totalnt)])
    B.totalnt = cat(2, B.totalnt - sum(B.classnt, 1), zeros(Int, 1, kk))
    B.classnt = randnumtable(B.α .* B.β[:,ones(Int, B.N0)]', B.C')
    B.totalnt = B.totalnt + sum(B.classnt, 1)

    # update beta weights
    a = zeros(B.K + 1)
    a[1:end-1] = B.totalnt
    a[end] = B.γ

    B.β = rand(Dirichlet(a))

    B
end

function updateenergy!(B::HDPData, X::Vector{Vector{T}}) where T <: Real

  E = 0
  numl = 0

  for j in length(X)

      # get samples of group
      data = X[j]

      pp = 0.0
      for i in length(data)
        for k = 1:length(B.distributions)
          pp += logpostpred( B.distributions[k], data[i] )[1] + log(B.weights[j][k])
        end
      end
      E += pp
      numl += 1

  end

  B.energy = E - log(numl)

end

function sampleparameters!(B::HDPBuffer, P::HDPHyperparam)
  totalnt = sum(B.classnt, 1)
  B.γ = resampleα(B.γ, sum(B.totalnt), B.K, k = P.γ_a, Θ = P.γ_b, maxiter = 10)
  B.α = resampleα(B.α, B.Nj, B.totalnt, k = P.α_a, Θ = P.α_b, maxiter = 10)
end

function extractpointestimate(B::HDPBuffer)
  W = map(j -> map(k -> B.C[k, j] + B.β[k] * B.α, 1:B.K), B.N0idx)
  model = HDPData(0.0, deepcopy(B.G), deepcopy(B.Z), deepcopy(W) )
  updateenergy!(model, B.X)
  model
end
