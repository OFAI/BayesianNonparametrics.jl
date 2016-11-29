using BayesianNonparametrics

(X, Y) = bloobs(randomize = false)

μ0 = vec(mean(X, 1))
κ0 = 5.0
ν0 = 9.0
Σ0 = cov(X)

H = WishartGaussian(μ0, κ0, ν0, Σ0)

model = DPM(H)
initialisation = KMeansInitialisation(k = 10)

modelBuffer = init(X, model, initialisation)
models = train(modelBuffer, DPMHyperparam(), Gibbs(maxiter = 500))

densities = Float64[m.energy for m in models]
activeComponents = Int[sum(m.weights .> 0) for m in models]
assignments = [m.assignments for m in models]

A = reduce(hcat, assignments)
(N, D) = size(X)

PSM = zeros(N, N)
M = size(A, 2)
for i in 1:N
  for j in 1:i
    PSM[i, j] = sum(A[i,:] .== A[j,:]) / M
    PSM[j, i] = sum(A[i,:] .== A[j,:]) / M
  end
end

mink = minimum([length(m.weights) for m in models])
maxk = maximum([length(m.weights) for m in models])

(peassignments, _) = pointestimate(PSM, method = :average, mink = mink, maxk = maxk)
