# Distributions

## Univariate Distributions

### Continuous

#### Normal with Normal-Gamma prior

Example:
```julia
H = GammaNormal(μ0 = 0.0, σ0 = 1.0, α0 = 1.0, β0 = 1.0)
```

#### Normal with Normal prior

Example:
```julia
H = NormalNormal(μ0 = 0.0, σ0 = 1.0)
```

### Discrete

#### Bernoulli with Beta prior

Example:
```julia
H = BetaBernoulli(α0 = 1.0, β0 = 1.0)
```

## Multivariate Distributions

### Continuous

#### Gaussian with Wishart prior

Example:
```julia
μ0 = zeros(5)
κ0 = 5.0
ν0 = 9.0
Σ0 = eye(5)
H = WishartGaussian(μ0, κ0, ν0, Σ0)
```

#### Gaussian with diagonal covariances

Example:
```julia
Hd = NormalNormal[NormalNormal() for d in 1:5]
H = GaussianDiagonal{NormalNormal}(Hd)
```

### Discrete

#### Multinomial with Dirichlet prior

 Example:
```julia
D = 5
α0 = 1.0
H = DirichletMultinomial(D, α0)
```