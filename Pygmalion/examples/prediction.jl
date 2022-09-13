function set_state_and_learnable_parameters!(mechanism, z, θ)
	# state parameters
	body = mechanism.bodies[1]
	parameters = get_parameters(body)
	parameters[1:6] .= z
	set_parameters!(body, parameters)

	# learnable parameters
	contact = mechanism.contacts[1]
	parameters = get_parameters(contact)
	parameters[1:1] .= θ # friction coefficient
	set_parameters!(contact, parameters)

	parameters = [get_parameters(body); get_parameters(contact)]
	mechanism.solver.parameters .= parameters
	return nothing
end

function prediction_loss(ẑ1, z1, z0, θ1, mechanism; complementarity_tolerance=1e-3)

	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance
	set_state_and_learnable_parameters!(mechanism, z0, θ1)

	u0 = zeros(mechanism.dimensions.input)
	w0 = nothing
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics(z1_pred, mechanism, z0, u0, w0)

	Q = I
	R = I
	l = 0.0
	l += 0.5 * (z1_pred - z1)' * Q * (z1_pred - z1)
	l += 0.5 * (z1 - ẑ1)' * R * (z1 - ẑ1)
	return l
end

function prediction_jacobian_state!(dz, z, θ, mechanism; complementarity_tolerance=1e-3)

	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance
	set_state_and_learnable_parameters!(mechanism, z, θ)

	u = zeros(mechanism.dimensions.input)
	w = nothing
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics_jacobian_state(dz, mechanism, z, u, w)
	get_next_state!(z1_pred, mechanism)
	return z1_pred
end

function prediction_jacobian_parameters!(dθ, z, θ, mechanism; complementarity_tolerance=1e-3)
	solver = mechanism.solver
	idx_learnable_parameters = 7:7

	solver.options.residual_tolerance = complementarity_tolerance / 10
	solver.options.complementarity_tolerance = complementarity_tolerance
	set_state_and_learnable_parameters!(mechanism, z, θ)

	dθsolver = zeros(mechanism.dimensions.state, solver.dimensions.parameters)
	u = zeros(mechanism.dimensions.input)
	w = nothing
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics_jacobian_parameters(dθsolver, mechanism, z, u, w)
	dθ .= dθsolver[:, idx_learnable_parameters]
	################################################################################################################################
	dθ .*= -10.0

	get_next_state!(z1_pred, mechanism)
	return z1_pred
end

# traj loss (z1:n, θ1:n)
function trajectory_loss(ẑ, z, θ, z0, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	l = 0.0
	z_prev = deepcopy(z0)
	for i = 1:H
		l += prediction_loss(ẑ[i], z[i], z_prev, θ[i], mechanism; complementarity_tolerance=complementarity_tolerance)
		z_prev .= z[i]
	end
	return l
end

# traj gradient wrt z1:n θ1:n
function trajectory_gradient(ẑ, z, θ, z0, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	nz = 6
	nθ = 1

	Q = I
	R = I
	dz = zeros(nz, nz)
	dθ = zeros(nz, nθ)
	grad = zeros(H * (nz + nθ))
	z_prev = deepcopy(z0)

	for i = 1:H
		z1_pred = prediction_jacobian_state!(dz, z_prev, θ[i], mechanism; complementarity_tolerance=complementarity_tolerance)
		z1_pred = prediction_jacobian_parameters!(dθ, z_prev, θ[i], mechanism; complementarity_tolerance=complementarity_tolerance)
		idx_zi = (i-1)*(nz+nθ) .+ (1:nz)
		idx_θi = (i-1)*(nz+nθ) + nz .+ (1:nθ)
		idx_zi_1 = (i-2)*(nz+nθ) .+ (1:nz)

		grad[idx_zi] .+= Q * (z[i] - z1_pred) + R * (z[i] - ẑ[i])
		grad[idx_θi] .+= -dθ' * Q * (z[i] - z1_pred)
		if i > 1
			grad[idx_zi_1] .+= -dz' * Q * (z[i] - z1_pred)
		end
		z_prev .= z[i]
	end
	# @warn "issue with grad sign on some part of the vector"
	return grad
end


# traj quasi newton hessian wrt z1:n θ1:n
function trajectory_hessian(ẑ, z, θ, z0, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	nz = 6
	nθ = 1

	Q = I(nz)
	R = I(nz)
	dz = zeros(nz, nz)
	dθ = zeros(nz, nθ)
	hess = spzeros(H * (nz + nθ), H * (nz + nθ))
	z_prev = deepcopy(z0)

	for i = 1:H
		idx_zi = (i-1)*(nz+nθ) .+ (1:nz)
		idx_θi = (i-1)*(nz+nθ) + nz .+ (1:nθ)
		idx_zi_1 = (i-2)*(nz+nθ) .+ (1:nz)

		z1_pred = prediction_jacobian_state!(dz, z_prev, θ[i], mechanism; complementarity_tolerance=complementarity_tolerance)
		z1_pred = prediction_jacobian_parameters!(dθ, z_prev, θ[i], mechanism; complementarity_tolerance=complementarity_tolerance)

		hess[idx_zi, idx_zi] .+= Q + R
		hess[idx_zi, idx_θi] .+= - Q * dθ
		hess[idx_θi, idx_zi] .+= - dθ' * Q

		hess[idx_θi, idx_θi] .+= dθ' * Q * dθ

		if i > 1
			hess[idx_zi, idx_zi_1] .+= - Q * dz
			hess[idx_zi_1, idx_zi] .+= - dz' * Q
			hess[idx_θi, idx_zi_1] .+= dθ' * Q * dz
			hess[idx_zi_1, idx_θi] .+= dz' * Q * dθ
			hess[idx_zi_1, idx_zi_1] .+= dz' * Q * dz
		end
		z_prev .= z[i]
	end
	return hess
end
