function vectorized_sdf(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix, A::AbstractVector,
		b::AbstractVector{Vector{T}}, δ) where T

	nβ = length(α) # number of rays
	np = length(A) # number of polytopes

	ϕ = zeros(T, nβ)
	for i in 1:np
		ϕi = vectorized_sdf(α, v, e, A[i], b[i], δ)
		ϕ = min.(ϕ, ϕi)
	end
	return ϕ
end

function vectorized_sdf(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix, A::AbstractMatrix{T},
		b::AbstractVector, δ) where T
	# α [nβ] vector of ray length
	# v [2 nβ] matrix holding the vector directions
	# A matrix half-spaces
	# b vector half-spaces
	# δ softness

	nβ = length(α) # number of rays
	nh = length(b) # number of half-spaces

	Av = A * v
	be = b .- A * e
	w = 1/δ * (α' .* Av .- be)
	wm = vec(findmax(w, dims=1)[1])
	ex = exp.(w .- wm')
	s = 1/nh * vec(sum(ex, dims=1)) # average
	ϕ = (log.(s) + wm) * δ
	return ϕ
end

function ray_rendering(α_prev::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
    A::AbstractMatrix, b::AbstractVector)

    # sdf nβ
    # sdfv nh x nβ
    # Av nh x nβ
    # αβ nβ
    # αhβ nh x nβ
    # α_ref nβ
    # v 2
    # A nh x 2
    # b nh

    nβ = length(α_prev)
    nh = length(b)

    αβ = α_prev
    Av = A * v
	be = b .- A * e
    αhβ = max.(0, Av .\ be) # we apply max to avoid intersection with object behind the camera.

    for i = 1:nh
        # for a given half-space i compute the value of the all half-spaces for all rays j
        sdfv = αhβ[i,:]' .* Av .- be
        # compute sdf for all rays j hitting the half-space i as the maximum over all half-spaces
        sdf = vec(findmax(sdfv, dims=1)[1])
        # compute condition for updating α
        cnd = (sdf .< 1e-5) .&& (αhβ[i,:] .< αβ)
        # update value of α if condition is met
        αβ = cnd .* αhβ[i,:] + (1 .- cnd) .* αβ
    end
    # we obtain the length of each ray after intersection with polytope A, b
    return αβ
end

function ray_rendering(v::AbstractMatrix{T}, e::AbstractMatrix,
    A::AbstractVector, b::AbstractVector; max_length=100.0) where T

    nβ = size(v, 2)
    np = length(b)

    αβ = max_length * ones(T,nβ)
    for i = 1:np
        αβ = ray_rendering(αβ, v, e, A[i], b[i])
    end

	return αβ
end

function rendering_loss(α_ref::AbstractVector, v::AbstractMatrix{T}, e::AbstractMatrix,
    A::AbstractVector, b::AbstractVector; max_length=100.0) where T

	αβ = ray_rendering(v, e, A, b, max_length=max_length)
	Δα = α_ref .- αβ
	l = sum(0.5 * Δα.^2 + abs.(Δα)) / nβ
	return l
end

function inside_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		inside_sample::Int=10,
		) where T

	nβ = length(α)

	# altitude (position along the z axis) of the point coming from ray j
	# altitude = α .* v[2,:] .+ e[2]
	samples = [(i-1) * thickness / (inside_sample - 1) for i = 1:inside_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .+ sample, v, e, A, b, δ_sdf)
		l += 10 * sum((1 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * inside_sample)
end

function outside_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		outside_sample::Int=10,
		) where T

	nβ = length(α)
	samples = [(i-1) * thickness / (outside_sample - 1) for i = 1:outside_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .- sample, v, e, A, b, δ_sdf)
		l += 5 * sum((0 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * outside_sample)
end

function floor_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		floor_sample::Int=10,
		) where T

	nβ = length(α)
	samples = [(i-1) * thickness / (floor_sample - 1) for i = 1:floor_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .+ sample, v, e, A, b, δ_sdf)
		l += 5 * sum((0 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * floor_sample)
end

function sdf_matching_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_softabs::T=0.5,
		) where T

	ϕ = vectorized_sdf(α, v, e, A, b, δ_sdf)
	l = 0.1 * sum(0.5 * ϕ.^2 + softabs.(ϕ, δ_softabs)) / nβ
	# l = 0.1 * (0.5 * ϕ.^2 + softabs.(ϕ, δ_softabs))
	# @warn "we need to remove the nβ factor"
	return l / nβ
end

function vectorized_ray(eye_positions::AbstractVector, β::AbstractVector, A::AbstractVector,
	 	b::AbstractVector, o::AbstractVector;
		altitude_threshold=0.01,
		max_length=100.0,
		)

	ne = length(eye_positions)
	np = length(A)

	bo = [b[j] + A[j] * o[j] for j = 1:np]

	α = []
	α_hit = []
	v = []
	v_hit = []
	e = []
	e_hit = []
	for i = 1:ne
		nβ = length(β[i])
		ei = hcat([eye_positions[i] for j = 1:nβ]...)
		vi = [cos.(β[i])'; sin.(β[i])']
		αi = ray_rendering(vi, ei, A, bo; max_length=max_length)
		# altitude (position along the z axis) of the point coming from ray j
		altitude = αi .* vi[2,:] .+ ei[2,:]
		cnd = altitude .>= altitude_threshold
		push!(α, αi)
		push!(α_hit, αi[cnd])
		push!(v, vi)
		push!(v_hit, vi[:,cnd])
		push!(e, ei)
		push!(e_hit, ei[:,cnd])
	end
	α = vcat(α...)
	α_hit = vcat(α_hit...)
	v = hcat(v...)
	v_hit = hcat(v_hit...)
	e = hcat(e...)
	e_hit = hcat(e_hit...)
	return α, α_hit, v, v_hit, e, e_hit
end

function keyword_shape_loss(α::AbstractVector, α_hit::AbstractVector,
		v::AbstractMatrix, v_hit::AbstractMatrix,
		e::AbstractMatrix, e_hit::AbstractMatrix,
		A::AbstractVector, b::AbstractVector, bo::AbstractVector,
		kwargs
		) where T
	return shape_loss(α, α_hit,
			v, v_hit,
			e, e_hit,
			A, b, bo;
			kwargs...
			)
end

function shape_loss(α::AbstractVector, α_hit::AbstractVector,
		v::AbstractMatrix, v_hit::AbstractMatrix,
		e::AbstractMatrix, e_hit::AbstractMatrix,
		A::AbstractVector, b::AbstractVector, bo::AbstractVector;
		δ_sdf=15.0,
		δ_softabs=0.5,
		δ_sigmoid=0.1,
		altitude_threshold=0.01,
		thickness=0.2,
		rendering=5.0,
		sdf_matching=20.0,
		overlap=2.0,
		individual=1.0,
		side_regularization=0.5,
		shape_regularization=0.5,
		inside=1.0,
		outside=0.1,
		floor=0.1,
		inside_sample::Int=10,
		outside_sample::Int=10,
		floor_sample::Int=10,
		) where T

	polytope_dimensions = length.(b)
	np = length(polytope_dimensions)
	# without the floor half-space
	Ar, br, bor = A[1:end-1], b[1:end-1], bo[1:end-1]
	l = 0.0
	# regularization
	for i = 1:np-1
		Δ = norm(br[i] .- mean(br[i]))
		l += side_regularization * 10.0 * (0.5*Δ^2 + softabs(Δ, δ_softabs)) / sum(polytope_dimensions)
	end

	# inside sampling, overlap penalty
	for i = 1:np
		p = o[i]
		ϕ = sum([sigmoid(-10*sdf(p, A[k], bo[k], zeros(2), δ_sdf)) for k in 1:np+1])
		l += overlap * 1e-2 * softplus(ϕ - 1, δ_softabs)^2 / np
		nh = polytope_dimensions[i]
		for j = 1:nh
			for α ∈ [1.00, 0.75, 0.5, 0.25]
				p = - α * A[i][j,:] .* bo[i][j] / norm(A[i][j,:])^2
				ϕ = sum([sigmoid(-10 * sdf(p, A[k], bo[k], zeros(2), δ_sdf)) for k in 1:np+1])
				l += overlap * 1e-2 * softplus(ϕ - 2, δ_softabs)^2 / (np * nh * length(α))
			end
		end
	end

	# regularization of polytope shape
	for i = 1:np
		nh = polytope_dimensions[i]
		for j = 1:nh
			p = - A[i][j,:] .* bo[i][j] / norm(A[i][j,:])^2
			ϕ = sdf(p, A[i], bo[i], zeros(2), δ_sdf)
			l += shape_regularization * 1.0 * (0.5*ϕ^2 + softabs(ϕ, δ_softabs)) / (np * nh)
		end
	end

	# rendering
	l += rendering * rendering_loss(α, v, e, A, bo; max_length=100.0)

	# # # individual
	# # l += individual * individual_loss(θ_f, polytope_dimensions_f, e[i], β[i], d_ref[i];
	# # 	δ_sdf=δ_sdf,
	# # 	δ_softabs=δ_softabs,
	# # 	altitude_threshold=altitude_threshold) / ne

	# sdf matching
	l += sdf_matching * sdf_matching_loss(α, v, e, A, b;
			δ_sdf=δ_sdf,
			δ_softabs=δ_softabs,
			)

	# floor sampling
	# only valid when the body frame is aligned with the world frame
	α_floor = - e_hit[2,:] ./ v_hit[2,:]
	l += floor * floor_loss(α_floor, v_hit, e_hit, A, bo;
			δ_sdf=δ_sdf,
			δ_sigmoid=δ_sigmoid,
			thickness=thickness,
			floor_sample=floor_sample,
			)

	# outside sampling
	l += outside * outside_loss(α_hit, v_hit, e_hit, A, bo;
			δ_sdf=δ_sdf,
			δ_sigmoid=δ_sigmoid,
			thickness=thickness,
			outside_sample=outside_sample,
			)

	# inside sampling
	l += inside * inside_loss(α_hit, v_hit, e_hit, A, bo;
			δ_sdf=δ_sdf,
			δ_sigmoid=δ_sigmoid,
			thickness=thickness,
			inside_sample=inside_sample,
			)
	return l
end
