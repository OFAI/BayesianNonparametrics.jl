# Distribution Functions

## Common Functions

#### log posterior predictive
The log posterior predictive can be computed using:

```julia
p = logpostpred(d::ConjugatePostDistribution, x)
```
resulting in a scalar if `x` and `d` is univariate or if `x` and `d` is multivariate. Otherwise, a vector is returned.

#### adding observations
Observations can be added to the conjugate posterior distribution using:

```julia
add!(d::ConjugatePostDistribution, x)
```

for inplace operations or using 

```julia
d = add(d::ConjugatePostDistribution, x)
```
resulting in a new distribution `d`.

#### removing observations
Observation can be removed from a conjugate posterior distribution using:

```julia
remove!(d::ConjugatePostDistribution, x)
```

for inplace operations of using

```julia
d = remove(d::ConjugatePostDistribution, x)
```
resulting in a new distribution `d`.

#### posterior parameters
The posterior parameters of a distribution can be obtained using:

```julia
parameters = posteriorParameters(d::ConjugatePostDistribution)
```
those parameters are returned as tuples, e.g. `parameters = (μ, σ)` in the case of NormalNormal distributions.
