function vectorized_sdf(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix, A::AbstractVector,
		b::AbstractVector{Vector{T}}, δ; max_length=50.00) where T

	nβ = length(α) # number of rays
	np = length(A) # number of polytopes

	ϕ = max_length * ones(T, nβ)
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

function ray_tracing(α_prev::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
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

function ray_tracing(v::AbstractMatrix{T}, e::AbstractMatrix,
    A::AbstractVector, b::AbstractVector; max_length=50.00) where T

    nβ = size(v, 2)
    np = length(b)

    αβ = max_length * ones(T,nβ)
    for i = 1:np
        αβ = ray_tracing(αβ, v, e, A[i], b[i])
    end

	return αβ
end

function rendering_loss(α_ref::AbstractVector, α_floor::AbstractVector, v::AbstractMatrix{T}, e::AbstractMatrix,
    A::AbstractVector, b::AbstractVector; max_length=50.00) where T

	αβ = ray_tracing(v, e, A, b, max_length=max_length)
	# since A, b does not contains the floor half-space we clip α to α_floor which is the maximum value of α with the floor half-space
	# this takes into accoun the fact that the polytope A, b, can be located at a non zero pose.
	α = min.(αβ, α_floor)
	Δα = α_ref .- α
	l = sum(0.5 * Δα.^2 + abs.(Δα)) / nβ
	return l
end

function inside_loss(α::AbstractVector,  v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		inside_sample::Int=10,
		max_length::T=50.00,
		) where T

	nβ = length(α)

	# altitude (position along the z axis) of the point coming from ray j
	# altitude = α .* v[2,:] .+ e[2]
	samples = [(i-1) * thickness / (inside_sample - 1) for i = 1:inside_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .+ sample, v, e, A, b, δ_sdf, max_length=max_length)
		l += 10 * sum((1 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * inside_sample)
end

function outside_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		max_length::T=50.00,
		outside_sample::Int=10,
		) where T

	nβ = length(α)
	samples = [(i-1) * thickness / (outside_sample - 1) for i = 1:outside_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .- sample, v, e, A, b, δ_sdf, max_length=max_length)
		l += 5 * sum((0 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * outside_sample)
end

function floor_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_sigmoid::T=0.1,
		thickness::T=0.2,
		max_length::T=50.00,
		floor_sample::Int=10,
		) where T

	nβ = length(α)
	samples = [(i-1) * thickness / (floor_sample - 1) for i = 1:floor_sample]

	l = 0.0
	for sample in samples
		ϕ = vectorized_sdf(α .+ sample, v, e, A, b, δ_sdf, max_length=max_length)
		l += 5 * sum((0 .- sigmoid.(-1/δ_sigmoid .* ϕ)).^2)
	end
	return l / (nβ * floor_sample)
end

function sdf_matching_loss(α::AbstractVector, v::AbstractMatrix, e::AbstractMatrix,
	    A::AbstractVector, b::AbstractVector;
		δ_sdf::T=15.0,
		δ_softabs::T=0.5,
		max_length::T=50.00,
		) where T

	ϕ = vectorized_sdf(α, v, e, A, b, δ_sdf, max_length=max_length)
	l = 0.1 * sum(0.5 * ϕ.^2 + softabs.(ϕ, δ_softabs))
	return l / nβ
end

Base.@kwdef mutable struct ShapeLossOptions1200{T}
	δ_sdf::T=0.025
	δ_sdf_matching::T=0.25
	δ_softabs::T=0.5
	δ_sigmoid::T=0.01
	altitude_threshold::T=0.01
	thickness::T=0.2
	max_length::T=50.00
	# rendering::T=5.0
	rendering::T=0.0
	sdf_matching::T=10.0
	side_regularization::T=0.5
	# inside::T=0.4
	inside::T=0.0
	outside::T=0.1
	floor::T=0.1
	inside_sample::Int=10
	outside_sample::Int=10
	floor_sample::Int=10
end

function shape_loss(α::AbstractVector, αmax::AbstractVector,
		v::AbstractMatrix, e::AbstractMatrix, hit_indices::AbstractVector,
		A::AbstractVector, b::AbstractVector, bo::AbstractVector; opts=ShapeLossOptions1200())

	polytope_dimensions = length.(b)
	np = length(polytope_dimensions)
	α_hit = α[hit_indices]
	αmax_hit = αmax[hit_indices]
	v_hit = v[:,hit_indices]
	e_hit = e[:,hit_indices]


	l = 0.0
	# regularization
	for i = 1:np
		Δ = norm(b[i] .- mean(b[i]))
		l += opts.side_regularization * 10.0 * (0.5*Δ^2 + softabs(Δ, opts.δ_softabs)) / sum(polytope_dimensions)
	end

	# rendering
	l += opts.rendering * rendering_loss(α, αmax, v, e, A, bo; max_length=opts.max_length)

	# sdf matching
	l += opts.sdf_matching * sdf_matching_loss(α_hit, v_hit, e_hit, A, bo;
			δ_sdf=opts.δ_sdf_matching,
			δ_softabs=opts.δ_softabs,
			max_length=opts.max_length,
			)

	# floor sampling
	l += opts.floor * floor_loss(αmax_hit, v_hit, e_hit, A, bo;
			δ_sdf=opts.δ_sdf,
			δ_sigmoid=opts.δ_sigmoid,
			thickness=opts.thickness,
			max_length=opts.max_length,
			floor_sample=opts.floor_sample,
			)

	# outside sampling
	l += opts.outside * outside_loss(α_hit, v_hit, e_hit, A, bo;
			δ_sdf=opts.δ_sdf,
			δ_sigmoid=opts.δ_sigmoid,
			thickness=opts.thickness,
			max_length=opts.max_length,
			outside_sample=opts.outside_sample,
			)

	# inside sampling
	l += opts.inside * inside_loss(α_hit, v_hit, e_hit, A, bo;
			δ_sdf=opts.δ_sdf,
			δ_sigmoid=opts.δ_sigmoid,
			thickness=opts.thickness,
			max_length=opts.max_length,
			inside_sample=opts.inside_sample,
			)
	return l
end

function noisy_shape_loss(
	α, αmax, v, e, hit_indices,
	A, b, bo,
	noise, denoise,
	opts=ShapeLossOptions1200())

	poses_noise = noise .- denoise
	v_noisy, e_noisy = noise_transform(v, e, poses_noise)
	l = shape_loss(
		α, αmax, v_noisy, e_noisy, hit_indices,
		A, b, bo,
		opts=opts,
		)
	H = length(denoise)
	l += 3e1 * sum(norm.(denoise).^2) / H
	return l
end

function noisy_shape_gradient(
	α, αmax, v, e, hit_indices,
	A, b, bo,
	noise, denoise,
	opts=ShapeLossOptions1200(),
	)

	gradient(noisy_shape_loss,
		α, αmax, v, e, hit_indices,
		A, b, bo,
		noise, denoise,
		opts,
		)[[6,7,8,10]]
end

function vectorized_ray(eye_positions::AbstractVector, angles::AbstractVector,
		A::AbstractVector, b::AbstractVector, o::AbstractVector,
		poses::AbstractVector=fill(zeros(3), length(eye_positions));
		altitude_threshold=0.01,
		max_length=50.00,
		)

	ne = length(eye_positions)
	np = length(A)

	# body to world frame transform
	x = [p[1:2] for p in poses] # position
	θ = [p[3] for p in poses] # orientation
	bRw = [[cos(θi) sin(θi); -sin(θi) cos(θi)] for θi in θ]
	# transform the eye_positions from world frame to body frame
	eye_positions_b = bRw .* (eye_positions .- x)
	# transform the angles from world frame to body frame
	angles_b = [angles[i] .- θ[i] for i=1:ne]
	# floor halfspace in the body frame
	Afw = [0.0 1.0;]
	bfw = [0.0]
	ofw = [0.0, 0.0]
	Afb = [Afw * bRw[i]' for i = 1:ne]
	bfb = [bfw for i = 1:ne]
	ofb = [bRw[i] * (ofw - x[i]) for i = 1:ne]

	bo = [b[j] + A[j] * o[j] for j = 1:np]

	α = []
	αmax = []
	v = []
	e = []
	hit_indices = []
	for i = 1:ne
		nβ = length(angles_b[i])
		ei = hcat([eye_positions_b[i] for j = 1:nβ]...)
		vi = [cos.(angles_b[i])'; sin.(angles_b[i])']
		Afi = [A..., Afb[i]]
		bofi = [bo..., bfb[i] + Afb[i] * ofb[i]]
		αi = ray_tracing(vi, ei, Afi, bofi; max_length=max_length)
		# we assume that the floor is located in 0,0 and with normal [0,1].
		αmaxi = min.(max_length, -eye_positions[i][2] ./ sin.(angles[i]))
		# altitude (position along the z axis in the world frame) of the point coming from ray j
		altitude = vec(Afb[i] * (αi' .* vi + ei .- ofb[i]) .- bfb[i])
		cnd = altitude .>= altitude_threshold
		push!(α, αi)
		push!(αmax, αmaxi)
		push!(v, vi)
		push!(e, ei)
		push!(hit_indices, cnd)
	end
	α = vcat(α...)
	αmax = vcat(αmax...)
	v = hcat(v...)
	e = hcat(e...)
	hit_indices = vcat(hit_indices...)
	return α, αmax, v, e, hit_indices
end

function noise_transform(v::AbstractMatrix, e::AbstractMatrix, poses_noise::AbstractVector)
	# poses_noise = poses_noisy .- poses
	# e and v are in the body frame
	# body ---- x θ ----> world <---- x_noisy, θ_noisy ---- body_noisy
	# We need to transform them to the noisy body frame.
	H = length(poses)
	nβ = Int(size(e, 2) / H)

	Δx1 = vec([poses_noise[i][1] for i = 1:H]' .* ones(nβ)) # nβ*H
	Δx2 = vec([poses_noise[i][2] for i = 1:H]' .* ones(nβ)) # nβ*H
	Δx = [Δx1'; Δx2'] # 2 x nβ*H
	Δθ = vec([poses_noise[i][3] for i = 1:H]' .* ones(nβ)) # nβ*H
	c = cos.(Δθ)
	s = sin.(Δθ)

	e_temp = e + Δx
	e1 = e_temp[1,:]
	e2 = e_temp[2,:]
	v1 = v[1,:]
	v2 = v[2,:]
	e_noisy = [(c .* e1)' .- (s .* e2)'; (s .* e1)' .+ (c .* e2)']
	v_noisy = [(c .* v1)' .- (s .* v2)'; (s .* v1)' .+ (c .* v2)']
	return v_noisy, e_noisy
end
