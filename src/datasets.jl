export bloobs

function bloobs(;centers = 3, samples = 100, randomize = true)

  μ = reduce(hcat, [rand(Uniform(-10, 10), centers) for i in 1:2])
  Σ = reduce(hcat, [rand(Uniform(0.5, 4), centers) for i in 1:2])

  samplespcenter = ones(Int, centers) * round(Int, samples / centers)

  for i = 1:(centers % samples)
      samplespcenter[i] += 1
  end

  X = reduce(hcat, [rand(MvNormal(ones(2) .* μ[i], eye(2) .* Σ[i]), samplespcenter[i]) for i in 1:centers])'
  Y = reduce(vcat, [ones(Int, samplespcenter[i]) * i for i in 1:centers])

  ids = collect(1:size(X, 1))
  if randomize
    shuffle!(ids)
  end

  X = X[ids,:]
  Y = Y[ids]

  return (X, Y)
end
