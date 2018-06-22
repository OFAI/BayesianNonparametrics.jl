export init, train

"abstract Hyperparameters"
abstract type AbstractHyperparam end;

"Abstract Model Data Object"
abstract type AbstractModelData end;

"Abstract Model Buffer Object"
abstract type AbstractModelBuffer end;

abstract type ModelType end;

abstract type InitialisationType end;

abstract type PosteriorInference end;

function init(X, model::ModelType, init::InitialisationType)
  throw(ErrorException("Initialisation $(typeof(init)) for $(typeof(model)) using $(typeof(X)) is not available."))
end

function extractpointestimate(B::AbstractModelBuffer)
  throw(ErrorException("No point estimate available for $(typeof(B))."))
end

function sampleparameters!(B::AbstractModelBuffer, P::AbstractHyperparam)
  throw(ErrorException("Parameter sampling for $(typeof(B)) using $(typeof(P)) is not available."))
end

function gibbs!(B::AbstractModelBuffer)
  throw(ErrorException("Gibbs sampling for $(typeof(B)) is not available."))
end

function slicesampling!(B::AbstractModelBuffer)
  throw(ErrorException("Slice sampling for $(typeof(B)) is not available."))
end

function variationalbayes!(B::AbstractModelBuffer)
  throw(ErrorException("Variational inference for $(typeof(B)) is not available."))
end

function train(B::AbstractModelBuffer, P::AbstractHyperparam, I::PosteriorInference)
  throw(ErrorException("Posterior inference for $(typeof(B)) with $(typeof(P)) and $(typeof(I)) is not available."))
end
