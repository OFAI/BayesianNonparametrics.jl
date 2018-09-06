using Statistics

(X, Y) = bloobs(randomize = false)

μ0 = vec(mean(X, dims = 1))
κ0 = 5.0
ν0 = 9.0
Σ0 = cov(X)

H = WishartGaussian(μ0, κ0, ν0, Σ0)

model = DPM(H)

initialisation = RandomInitialisation(k = 10)
modelBuffer = init(X, model, initialisation)

model0 = BayesianNonparametrics.extractpointestimate(modelBuffer)
model1 = train(modelBuffer, DPMHyperparam(), Gibbs(maxiter = 1))[end]

initialisation = KMeansInitialisation(k = 10)
modelBuffer = init(X, model, initialisation)

model0 = BayesianNonparametrics.extractpointestimate(modelBuffer)
model1 = train(modelBuffer, DPMHyperparam(), Gibbs(maxiter = 1))[end]
