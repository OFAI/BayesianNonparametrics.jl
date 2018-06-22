# Utility Functions

[\< back](README.md)

#### Resample Concentration Parameter
The concentration parameter using in Dirichlet process models can be resampled using the approach by Escobar and West using:

```julia
resampleα(α::Float64, N::Int, K::Int)
```
where `N` is the number of observations and `K` the number of clusters.

**Optional Arguments**
```julia
k = 1.0 # shape of Gamma distribution
Θ = 1.0 # scale of Gamma distribution
maxiter = 1 # number of iterations of the sampling proceedure
```

#### Randomly select an Index
We can randomly draw an index from a probability table using:

```julia
index::Int = randomindex(p::Vector)
```
the `index` will be drawn proportional to the un-normalized probabilities in `p`.
