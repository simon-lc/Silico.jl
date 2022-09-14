################################################################################
# initialization
################################################################################
function parameter_initialization(d::Matrix, polytope_dimensions)
	np = length(polytope_dimensions)

	# k-mean clustering
	kmres = Clustering.kmeans(d, np)
	# initialization
	b_mean = 2 * mean(sqrt.(kmres.costs))
	θ = zeros(0)
	for i = 1:np
	    angles = range(-π, π, length=nh+1)[1:end-1]
		θi = [vcat([[cos(a), sin(a)] for a in angles]...); b_mean*ones(nh); kmres.centers[:, i]]
	    A, b, o = unpack_halfspaces(θi)
	    push!(θ, pack_halfspaces(A, b, o)...)
	end

	return θ, kmres
end

function filter_point_cloud(d::Vector; poses=fill(zeros(3), length(d)), altitude_threshold=0.01)
	ne = length(d)
	# point cloud reaching the object
	d_object = []

	for i = 1:ne
		p = poses[i][1:2]
		θ = poses[i][3]
		bRw = [cos(θ) sin(θ); -sin(θ) cos(θ)]
		nβ = size(d[i],2)
		for j = 1:nβ
		    d_world = d[i][:,j]
			if d_world[2] > altitude_threshold
				d_body = bRw * (d_world - p)
		        push!(d_object, d_body)
		    end
		end
	end
	d_object = hcat(d_object...)
	return d_object
end


################################################################################
# projection
################################################################################
function projection(θ, polytope_dimensions;
		Alims=[-1.00, +1.00],
		blims=[+0.05, +0.40],
		olims=[-3.00, +3.00],
		)
	θmin = []
	θmax = []
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)

    for (i,nh) in enumerate(polytope_dimensions)
		for j = 1:nh
			A[i][j,:] = A[i][j,:] / (1e-6 + norm(A[i][j,:]))
		end
		push!(θmin, [Alims[1] * ones(2nh); blims[1] * ones(nh); olims[1] * ones(2)]...)
		push!(θmax, [Alims[2] * ones(2nh); blims[2] * ones(nh); olims[2] * ones(2)]...)
    end

    return clamp.(θ, θmin, θmax)
end


################################################################################
# step_projection
################################################################################
function step_projection(Δθ, polytope_dimensions;
		Alims=[-0.60, +0.60],
		blims=[-0.05, +0.05],
		olims=[-0.05, +0.05],
		)
	Δθmin = []
	Δθmax = []
	for nh in polytope_dimensions
		push!(Δθmin, [Alims[1] * ones(2nh); blims[1] * ones(nh); olims[1] * ones(2)]...)
		push!(Δθmax, [Alims[2] * ones(2nh); blims[2] * ones(nh); olims[2] * ones(2)]...)
	end
    return clamp.(Δθ, Δθmin, Δθmax)
end

################################################################################
# loss
################################################################################
function inside_loss(θ::Vector{D}, polytope_dimensions::Vector{Int},
	e::Vector{T}, β, d_ref::Matrix{T};
	δ_sdf::T=75.0,
	δ_softabs::T=0.01,
	δ_sigmoid::T=0.1,
	altitude_threshold::T=0.01,
	thickness::T=0.2,
	inside_sample::Int=10,
	) where {T,D}

	nβ = length(β)
	l = 0.0

	for i = 1:nβ
		@views p_ref = d_ref[:,i]
		v_ref = SVector{2,T}(cos(β[i]), sin(β[i]))
		if p_ref[2] >= altitude_threshold
			# sample inside points for a distance equal to thickness
			α_ref = p_ref' * v_ref - e' * v_ref
			for α in range(α_ref, α_ref + thickness, length=inside_sample)
				p = e + α * v_ref
				ϕ = sdfV(p, θ, polytope_dimensions, δ_sdf)
				l += 10 * (1 - sigmoid(-1/δ_sigmoid * ϕ))^2
			end
		end
	end
	return l / (nβ * inside_sample)
end

function outside_loss(θ::Vector{D}, polytope_dimensions::Vector{Int},
	e::Vector{T}, β, d_ref::Matrix{T};
	δ_sdf::T=75.0,
	δ_softabs::T=0.01,
	δ_sigmoid::T=0.1,
	altitude_threshold::T=0.01,
	thickness::T=0.2,
	outside_sample::Int=10,
	) where {T,D}

	nβ = length(β)
	l = 0.0

	for i = 1:nβ
		@views p_ref = d_ref[:,i]
		if p_ref[2] >= altitude_threshold
			# sample outside points for a distance equal to thickness before we reach the actual point
			v_ref = SVector{2,T}(cos(β[i]), sin(β[i]))
			α_ref = p_ref' * v_ref - e' * v_ref
			α_max = max(0, α_ref - thickness)
			for α in range(α_max, α_ref, length=outside_sample)
				p = e + α * v_ref
				ϕ = sdfV(p, θ, polytope_dimensions, δ_sdf)
				l += 5 * (0 - sigmoid(-1/δ_sigmoid * ϕ))^2
			end
		end
	end

	return l / (nβ * outside_sample)
end

function floor_loss(θ::Vector{D}, polytope_dimensions::Vector{Int},
	e::Vector{T}, β, d_ref::Matrix{T};
	δ_sdf::T=75.0,
	δ_softabs::T=0.01,
	δ_sigmoid::T=0.1,
	altitude_threshold::T=0.01,
	thickness::T=0.2,
	floor_sample::Int=10,
	) where {T,D}

	nβ = length(β)
	l = 0.0

	for i = 1:nβ
		@views p_ref = d_ref[:,i]
		if p_ref[2] >= altitude_threshold
			# sample outside points for a distance equal to thickness before we reach the actual point
			v_ref = SVector{2,T}(cos(β[i]), sin(β[i]))
			α_ref = p_ref' * v_ref - e' * v_ref
			α_floor = - e[2] / (1e-6 + v_ref[2])
			for α in range(α_floor, α_floor + thickness, length=floor_sample)
				p = e + α * v_ref
				ϕ = sdfV(p, θ, polytope_dimensions, δ_sdf)
				l += 5 * (0 - sigmoid(-1/δ_sigmoid * ϕ))^2
			end
		end
	end

	return l / (nβ * floor_sample)
end

function sdf_matching_loss(θ::Vector{D}, polytope_dimensions::Vector{Int},
	e::Vector{T}, β, d_ref::Matrix{T};
	δ_sdf::T=75.0,
	δ_softabs::T=0.01,
	) where {T,D}

	nβ = length(β)
	l = 0.0

	for i = 1:nβ
		@views p = d_ref[:,i]
		ϕ = sdfV(p, θ, polytope_dimensions, δ_sdf)
		l += 0.1 * (0.5*ϕ^2 + softabs(ϕ, δ_softabs)) / nβ
	end
	return l / nβ
end

function individual_loss(θ::Vector{D}, polytope_dimensions::Vector{Int},
	e::Vector{T}, β, d_ref::Matrix{T};
	δ_sdf::T=75.0,
	δ_softabs::T=0.01,
	altitude_threshold=0.01,
	) where {T,D}

	nβ = length(β)
	np = length(polytope_dimensions)
	l = 0.0

	for i = 1:nβ
		@views p = d_ref[:,i]
		if p[2] > altitude_threshold
			idx = 0
			min_distance = +Inf
			for j = 1:np
				Aj, bj, oj = unpack_halfspaces(θ, polytope_dimensions, j)
				distance = norm(oj - p)
				if distance < min_distance
					idx = j
					min_distance = distance
				end
			end
			Aidx, bidx, oidx = unpack_halfspaces(θ, polytope_dimensions, idx)
			ϕ = sdfV(p, Aidx, bidx, oidx, δ_sdf)
			l += 10.0 * (0.5*ϕ^2 + softabs(ϕ, δ_softabs))
		end
	end
	return l / nβ
end

function shape_loss(θ, polytope_dimensions, e, β, ρ, d_ref;
	altitude_threshold=0.01,
	thickness=0.2,
	δ_sdf=75.0,
	δ_softabs=0.01,
	δ_sigmoid=0.1,
	rendering=10.0,
	add_rendering=true,
	sdf_matching=1.0,
	overlap=0.1,
	individual=1.0,
	side_regularization=2.0,
	shape_regularization=1.0,
	inside=1.0,
	outside=1.0,
	floor=1.0,

	inside_sample::Int=10,
	outside_sample::Int=10,
	floor_sample::Int=10,
	)

	np = length(polytope_dimensions)
	ne = length(e)
	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	A_f, b_f, o_f = unpack_halfspaces(θ_f, polytope_dimensions_f)

	l = 0.0
	# regularization
	l += side_regularization * 10.0 * sum([0.5*norm(bi .- mean(bi))^2 + softabs(norm(bi .- mean(bi)), δ_softabs) for bi in b]) / (np * nh)

	# inside sampling, overlap penalty
	for i = 1:np
		p = o[i]
		ϕ = sum([sigmoid(-10*sdf(p, A_f[k], b_f[k], o_f[k], δ_sdf)) for k in 1:np+1])
		l += overlap * 1e-2 * softplus(ϕ - 1, δ_softabs)^2 / np
		nh = polytope_dimensions[i]
		for j = 1:nh
			for α ∈ [1.00, 0.75, 0.5, 0.25]
				p = o[i] - α * A[i][j,:] .* b[i][j] / norm(A[i][j,:])^2
				ϕ = sum([sigmoid(-10 * sdf(p, A_f[k], b_f[k], o_f[k], δ_sdf)) for k in 1:np+1])
				l += overlap * 1e-2 * softplus(ϕ - 2, δ_softabs)^2 / (np * nh * length(α))
			end
		end
	end

	# regularization of polytope shape
	for i = 1:np
		nh = polytope_dimensions[i]
		for j = 1:nh
			p = o[i] - A[i][j,:] .* b[i][j] / norm(A[i][j,:])^2
			ϕ = sdf(p, A[i], b[i], o[i], δ_sdf)
			# l += shape_regularization * 10 * (1 - sigmoid(-1/δ_sigmoid * ϕ))^2 / (np * nh)
			# l += shape_regularization * 10 * (1 - sigmoid(-1/δ_sigmoid * ϕ))^2 / (np * nh)
			l += shape_regularization * 1.0 * (0.5*ϕ^2 + softabs(ϕ, δ_softabs)) / (np * nh)
		end
	end

	for i = 1:ne
		# rendering
		add_rendering && (l += rendering * trans_point_loss(e[i], β[i], ρ, θ_f, polytope_dimensions_f, d_ref[i]))

		# individual
		l += individual * individual_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs,
			altitude_threshold=altitude_threshold) / ne

		# sdf matching
		l += sdf_matching * sdf_matching_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs) / ne

		# floor sampling
		l += floor * floor_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs,
			δ_sigmoid=δ_sigmoid,
			altitude_threshold=altitude_threshold,
			thickness=thickness,
			floor_sample=floor_sample) / ne

		# outside sampling
		l += outside * outside_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs,
			δ_sigmoid=δ_sigmoid,
			altitude_threshold=altitude_threshold,
			thickness=thickness,
			outside_sample=outside_sample) / ne

		# inside sampling
		l += inside * inside_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs,
			δ_sigmoid=δ_sigmoid,
			altitude_threshold=altitude_threshold,
			thickness=thickness,
			inside_sample=inside_sample) / ne
	end
	return l
end

shape_grad(θ, polytope_dimensions, e, β, ρ, d; parameters...) =
	ForwardDiff.gradient(θ -> shape_loss(θ, polytope_dimensions, e, β, ρ, d; parameters...), θ)

function shape_hess(θ, polytope_dimensions, e, β, ρ, d; parameters...)
	nθ = length(θ)
	H = spzeros(nθ, nθ)

	off = 0
	for nh in polytope_dimensions
		ind = off .+ (1:nh)
		off += nh
		function local_loss(θi::Vector{T}) where T
			θl = zeros(T, nθ)
			θl .= θ
			θl[ind] .= θi
			shape_loss(θl, e, β, ρ, d; parameters..., add_rendering=false)
		end
		H[ind, ind] .= ForwardDiff.hessian(θi -> local_loss(θi), θ[ind])
	end

	return H
end







#
#
#
#
#
#
#
#
#
# individual_parameters = Dict(
# 	:δ_sdf => 15.0,
# 	:δ_softabs => 0.01,
# )
#
# ep = [e0]
# βp = [β0]
# dp = [d0]
# outside_loss(θ, polytope_dimensions, ep, βp, dp; individual_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, ep, βp, dp; individual_parameters...) for i=1:100]
# @benchmark outside_loss($θ, $polytope_dimensions, $ep, $βp, $dp)
#
# outside_loss(θ, polytope_dimensions, e0, β0, d0; individual_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, e0, β0, d0; individual_parameters...) for i=1:100]
# @benchmark outside_loss($θ, $polytope_dimensions, $e0, $β0, $d0)
#
#
# sdf_matching_parameters = Dict(
# 	:δ_sdf => 15.0,
# 	:δ_softabs => 0.01,
# )
#
# ep = [e0]
# βp = [β0]
# dp = [d0]
# outside_loss(θ, polytope_dimensions, ep, βp, dp; sdf_matching_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, ep, βp, dp; sdf_matching_parameters...) for i=1:100]
# @benchmark outside_loss($θ, $polytope_dimensions, $ep, $βp, $dp)
#
# outside_loss(θ, polytope_dimensions, e0, β0, d0; sdf_matching_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, e0, β0, d0; sdf_matching_parameters...) for i=1:1000]
# @benchmark outside_loss($θ, $polytope_dimensions, $e0, $β0, $d0)
#
#
#
# outside_parameters = Dict(
# 	:δ_sdf => 15.0,
# 	:δ_softabs => 0.01,
# 	:δ_sigmoid => 0.1,
# 	:altitude_threshold => 0.01,
# 	:thickness => 0.2,
# 	:outside_sample => 10,
# )
#
# ep = [e0]
# βp = [β0]
# dp = [d0]
# outside_loss(θ, polytope_dimensions, ep, βp, dp; outside_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, ep, βp, dp; outside_parameters...) for i=1:100]
# @benchmark outside_loss($θ, $polytope_dimensions, $ep, $βp, $dp)
#
# outside_loss(θ, polytope_dimensions, e0, β0, d0; outside_parameters...)
# Main.@profiler [outside_loss(θ, polytope_dimensions, e0, β0, d0; outside_parameters...) for i=1:1000]
# @benchmark outside_loss($θ, $polytope_dimensions, $e0, $β0, $d0)
#
#
#
#
#
#
#
#
#
# function inside_loss(θ_f::Vector{D}, polytope_dimensions_f::Vector{Int},
# 		e::Vector, β::Vector, d_ref::Vector{Matrix{T}}; kwargs...) where {T,D}
# 	ne = length(e)
#
# 	l = 0.0
# 	for i = 1:ne
# 		l += inside_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i]; kwargs...)
# 	end
# 	return l / ne
# end
#
# function inside_loss(θ_f::Vector{D}, polytope_dimensions_f::Vector{Int},
# 	e::Vector{T}, β, d_ref::Matrix{T};
# 	δ_sdf::T=75.0, # 	δ_softabs::T=0.01,
# 	δ_sigmoid::T=0.1,
# 	altitude_threshold::T=0.01,
# 	thickness::T=0.2,
# 	inside_sample::Int=10,
# 	) where {T,D}
#
# 	nβ = length(β)
# 	l = 0.0
#
# 	for j = 1:nβ
# 		@views p_ref = d_ref[:,j]
# 		v_ref = SVector{2,T}(cos(β[j]), sin(β[j]))
# 		if p_ref[2] >= altitude_threshold
# 			# sample inside points for a distance equal to thickness
# 			α_ref = p_ref' * v_ref - e' * v_ref
# 			for α in range(α_ref, α_ref + thickness, length=inside_sample)
# 				p = e + α * v_ref
# 				ϕ = sdfV(p, θ_f, polytope_dimensions_f, δ_sdf)
# 				l += 10 * (1 - sigmoid(-1/δ_sigmoid * ϕ))^2
# 			end
# 		end
# 	end
# 	return l / (nβ * inside_sample)
# end
#
#
# inside_parameters = Dict(
# 	:δ_sdf => 15.0,
# 	:δ_softabs => 0.01,
# 	:δ_sigmoid => 0.1,
# 	:altitude_threshold => 0.01,
# 	:thickness => 0.2,
# 	:inside_sample => 10,
# )
#
# ep = [e0]
# βp = [β0]
# dp = [d0]
# inside_loss(θ, polytope_dimensions, ep, βp, dp; inside_parameters...)
# Main.@profiler [inside_loss(θ, polytope_dimensions, ep, βp, dp; inside_parameters...) for i=1:100]
# @benchmark inside_loss($θ, $polytope_dimensions, $ep, $βp, $dp)
#
# inside_loss(θ, polytope_dimensions, e0, β0, d0; inside_parameters...)
# Main.@profiler [inside_loss(θ, polytope_dimensions, e0, β0, d0; inside_parameters...) for i=1:1000]
# @benchmark inside_loss($θ, $polytope_dimensions, $e0, $β0, $d0)
#
#
#
#
#
#
# δp1 = 0.01
# p1 = [1.0, 0.0]
# δp1 = 0.01
# Afp1 = reshape(Ap1, 8)
#
# polytope_dimensions = [4,4,4]
# Afp = [reshape(Ap2, 8); reshape(Ap1, 8); reshape(Ap2, 8)]
# bp = [bp0; bp1; bp2]
# op = [op0; op1; op2]
# θ = rand(3 * (3 * 4  + 2))
#
# sdf(p1, Ap1, bp1, op1, δp1)
# sdfV(p1, Afp1, bp1, op1, δp1)
# sdfV(p1, θ, polytope_dimensions, δp1)
#
# unpack_halfspaces(θ, polytope_dimensions, 2)
# @benchmark unpack_halfspaces($θ, $polytope_dimensions, $2)
#
#
#
# @benchmark sdf($p1, $Ap1, $bp1, $op1, $δp1)
# @benchmark sdfV($p1, $Afp1, $bp1, $op1, $δp1)
# @benchmark sdfV1($p1, $θ, $polytope_dimensions, $δp1)
