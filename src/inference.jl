export Gibbs, SliceSampler

type Gibbs <: PosteriorInference
        burnin::Int
        thinout::Int
        maxiter::Int

        Gibbs(; burnin = 0, thinout = 1, maxiter = 100) = new(burnin, thinout, maxiter)
end

type SliceSampler <: PosteriorInference
        burnin::Int
        thinout::Int
        maxiter::Int

        SliceSampler(; burnin = 0, thinout = 1, maxiter = 100) = new(burnin, thinout, maxiter)
end

function train(B::AbstractModelBuffer, P::AbstractHyperparam, I::Gibbs)

  results = AbstractModelData[]

  # burn-in phase
  for iter in 1:I.burnin
    # run one gibbs iteration
    gibbs!(B)

    # sample parameters
    sampleparameters!(B, P)
  end

  # Gibbs sweeps
  @showprogress 1 "Running Gibbs..." for iter in 1:I.maxiter
    for t in 1:I.thinout
      # run one gibbs iteration
      gibbs!(B)

      # sample parameters
      sampleparameters!(B, P)
    end

    push!(results, extractpointestimate(B))
  end

  return results

end

function train(B::AbstractModelBuffer, P::AbstractHyperparam, I::SliceSampler)

  results = AbstractModelData[]

  # burn-in phase
  for iter in 1:I.burnin
    # run one gibbs iteration
    slicesampling!(B)

    # sample parameters
    sampleparameters!(B, P)
  end

  # Gibbs sweeps
  @showprogress 1 "Running SliceSampler..." for iter in 1:I.maxiter
    for t in 1:I.thinout
      # run one gibbs iteration
      slicesampling!(B)

      # sample parameters
      sampleparameters!(B, P)
    end

    push!(results, extractpointestimate(B))
  end

  return results

end
