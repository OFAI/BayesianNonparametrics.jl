using LinearAlgebra
# Testing the following distribution: WishartGaussian, DirichletMultinomial, GammaNormal, NormalNormal, BetaBernoulli

@testset "testing posterior paramters" begin

  @testset "Normal-Inverse-Wishart" begin

    # prior
    μ0 = zeros(2)
    κ0 = 5.0
    ν0 = 9.0
    Σ0 = Matrix{Float64}(I, 2, 2) 

    # data
    x = [0.2 0.1; 0.1 0.0; 0.2 0.05]
    (N, D) = size(x)

    # distribution
    d = WishartGaussian(μ0, κ0, ν0, Σ0)

    # test prior
    (μ, κ, ν, Σ) = BayesianNonparametrics.posteriorParameters(d)

    @test κ == κ0
    @test ν == ν0
    @test all( μ .== μ0 )
    @test all( Σ .== Σ0 )

    # add data
    BayesianNonparametrics.add!(d, x)

    # test posterior parameters
    (μ, κ, ν, Σ) = BayesianNonparametrics.posteriorParameters(d)

    mu = vec(mean(x, dims=1))
    SS = x' * x

    @test κ == κ0 + N
    @test ν == ν0 + N
    @test all( μ .≈ (κ0 * μ0 .+ mu * N) ./ κ )
    @test all( Σ .== Σ0 + SS - κ * (μ * μ') + κ0 * (μ0 * μ0') )

    # remove data
    BayesianNonparametrics.remove!(d, x)

    # test prior
    (μ, κ, ν, Σ) = BayesianNonparametrics.posteriorParameters(d)

    @test κ == κ0
    @test ν == ν0
    @test all( μ .== μ0 )
    @test all( Σ .== Σ0 )
  end

  @testset "Normal-Gamma" begin

    # prior
    μ0 = 0.0
    λ0 = 1.0
    α0 = 1.0
    β0 = 1.0

    # data
    x = [0.1, 0.0, 0.05]
    N = length(x)

    # distribution
    d = GammaNormal(μ0, λ0, α0, β0)

    # test prior
    (μ, λ, α, β) = BayesianNonparametrics.posteriorParameters(d)

    @test μ == μ0
    @test λ == λ0
    @test α == α0
    @test β == β0

    # add data
    BayesianNonparametrics.add!(d, x)

    # test posterior paramters
    (μ, λ, α, β) = BayesianNonparametrics.posteriorParameters(d)

    #@test μ ==    
    @test λ == λ0 + N
    @test α = α0 + (N / 2)
    #@test β == 

    # remove data
    BayesianNonparametrics.remove!(d, x)

    # test prior
    (μ, λ, α, β) = BayesianNonparametrics.posteriorParameters(d)

    @test μ == μ0
    @test λ == λ0
    @test α == α0
    @test β == β0
  end

end
