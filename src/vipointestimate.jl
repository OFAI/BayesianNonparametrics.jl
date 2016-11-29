export pointestimate

"""

	point_estimate_avg(psm::Array{Float64, 2})

Find optimal partition which minimizes the lower bound to the Variation of Information
obtain from Jensen's inequality where the expectation and log are reversed.

Code based on R implementation by Sara Wade <sara.wade@eng.cam.ac.uk>
"""
function point_estimate_hclust(psm::Array{Float64, 2}; maxk = -1, mink = mink, method = :average)

	havg = hclust(1-psm, method)
	cls = reduce(hcat, map(k -> cutree(havg; k = k), mink:maxk))

	# compute Variation of Information lower bound
	vi = variation_of_information_lb(cls, psm)

	idx = findfirst(minimum(vi) .== vi)

	if ndims(cls) == 2
		cl = cls[:,idx]
	else
		cl = cls
	end

	return (cl, minimum(vi))

end

"""
	point_estimate(psm::Array{Float64, 2})

Find optimal partition which minimizes the lower bound to the Variation of Information
obtain from Jensen's inequality where the expectation and log are reversed.

Code based on R implementation by Sara Wade <sara.wade@eng.cam.ac.uk>
"""
function pointestimate(psm::Array{Float64, 2}; method = :avg, maxk = -1, mink = 2)

	methods = [:average, :complete, :greedy]
	@assert method in methods "Method must be in $(methods)."

	if maxk == -1
		maxk = convert(Int, ceil(size(psm)[1] / 4.0))
	end

	if method in [:average, :complete]
		return point_estimate_hclust(psm, maxk = maxk, mink = mink, method = method)
	else
		println("not implemented, yet")
	end
end

"""
	variation_of_information_lb()
Computes the lower bound to the posterior expected Variation of Information.
"""
function variation_of_information_lb(cls::Array{Int, 2}, psm)

	N = size(psm)[1]
	F = zeros(size(cls)[2])

	for ci in 1:size(cls)[2]
		c = cls[:,ci]
		for i in 1:N
			ind = c .== c[i]
			F[ci] += (log2(sum(ind)) +log2(sum(psm[i,:])) -2 * log2(sum(ind' .* psm[i,:]))) / N
		end
	end

	return F

end

"""
	variation_of_information_lb()
Computes the lower bound to the posterior expected Variation of Information.
"""
function variation_of_information_lb(cls::Vector{Int}, psm)

	N = size(psm)[1]
	F = zeros(1)

	for i in 1:N
		ind = cls .== cls[i]
		F[1] += (log2(sum(ind))+log2(sum(psm[i,:]))-2*log2(sum(ind' .* psm[i,:]))) / N
	end

	return F

end
