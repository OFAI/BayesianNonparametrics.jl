export resampleα, randomindex

"""
	resampleα(α::Float64, N::Int, K::Int)

Resample α given the approach by Escobar and West. (page 585)
"""
function resampleα(α::Float64, N::Int, K::Int; k = 1.0, Θ = 1.0, maxiter = 1)

    w = zeros(2)
    w[2] = N

    f0 = k + K
    f1 = Θ + K - 1

    newα = α

    for i in 1:maxiter

        η = rand(Beta(newα + 1, N))
        f3 = Θ - log(η)

        w[1] = f1 / f3

        z = randomindex(w)

        if z == 1
            newα = rand(Gamma(f0, 1/f3))
        else
            newα = rand(Gamma(f1, 1/f3))
        end
    end

    return newα
end

"""
	randomindex(p::Vector{Float64})

Randomly select an index propotional to its probability.
"""
function randomindex(p::Vector{Float64})

    max = sum(p)
    csum = 0.0
    thres = rand()

    @inbounds for i in 1:length(p)
        csum += p[i]
        if csum >= thres * max
            return convert(Int, i)
        end
    end

    return length(p)
end
