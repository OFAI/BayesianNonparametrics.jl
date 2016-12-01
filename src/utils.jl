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

function resampleα(α::Float64, N::Array{Int}, K::Array{Int}; k = 1.0, Θ = 1.0, maxiter = 1)
    # implementation according to escobar - west page
    totalK = sum(K)
    num = length(N)

    η = zeros(num)

		newα = α
    for i in 1:maxiter

        for j in 1:num
            η[j] = rand(Beta(newα + 1, N[j]))
        end

        z = rand(num) .* (newα + N) .< N

        g_a = k + totalK - sum(z)
        g_b = Θ - sum(log(η))

        newα = rand(Gamma(g_a, 1/g_b))
    end

    newα
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

function randnumtable(weights::Array{Float64}, table::Array{Int})

    numtable = zeros(Int, size(table))

    B = unique( table )
    w = log( weights )

    for i in 1:length(B)

        max = B[i]
        if max > 0

            m = 1:max

            stirnums = map( x -> abs(stirlings1(max, x)), m)
            stirnums /= maximum(stirnums)

            for (idx, j) in enumerate(find(table .== max))

                clike = m .* w[idx]
                clike = cumsum(stirnums .* exp(clike - maximum(clike)))

                numtable[j] = 1+sum(rand() * clike[max] .> clike)
            end


        end

    end

    return numtable
end
