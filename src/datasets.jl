export bloobs, bars

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

function bars(;img_size = 6, noise_level = 1e-4, num_per_mixture = [0 ones(1,3) / 3], num_group = 20, num_data = 50)

  numdim   = img_size^2;
  numbars  = img_size*2

  mix_theta = zeros(numdim, numbars)

  # add horizontal bars
  for i in 1:img_size
    img = ones(img_size, img_size) .* noise_level / img_size^2;
    img[:,i] = img[:,i] + 1/img_size;
    mix_theta[:,i] = img[:] / sum(img[:]);
  end

  # add vertical bars
  for i = 1:img_size
    img = ones(img_size, img_size) .* noise_level / img_size^2;
    img[i,:] = img[i,:] + 1/img_size;
    mix_theta[:,img_size+i] = img[:]/sum(img[:]);
  end

  # generate samples
  # Note: strange behavior of cumsum!!!
  cumbar = cumsum(num_per_mixture, 2)

  samples = Vector{Vector{Int}}(num_group)

  for g in 1:num_group

      # random number of mixture components
      n_bars = 1 + sum(rand() .> cumbar)
      k = randperm(numbars)[1:n_bars]

      # get weigths
      theta = mean(mix_theta[:,k], 2)

      # get samples (e.g. words)
      samples[g] = vec( 1 + sum(repmat(rand(num_data)', numdim, 1) .> repmat(cumsum(theta, 1), 1, num_data), 1) )
  end

  # return samples and bars
  return (samples, mix_theta)

end
