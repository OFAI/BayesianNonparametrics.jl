using BayesianNonparametrics

img_size = 3
(X, trueWeights) = bars(img_size = img_size)

D = img_size^2
H = DirichletMultinomial(D, 1.0)

model = HDP(H)
initialisation = RandomInitialisation(k = 20)

modelBuffer = init(X, model, initialisation)
model0 = BayesianNonparametrics.extractpointestimate(modelBuffer)

models = train(modelBuffer, HDPHyperparam(), Gibbs(maxiter = 50))

densities = Float64[m.energy for m in models]
activeComponents = Int[length(m.distributions) for m in models]

println(densities)
