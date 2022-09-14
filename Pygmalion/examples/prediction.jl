function prediction_loss(ẑ1, z1, z0, w1, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance

	u0 = zeros(mechanism.dimensions.input)
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics(z1_pred, mechanism, z0, u0, w=w1, idx_parameters=idx_parameters)

	Q = I
	R = I
	l = 0.0
	l += 0.5 * (z1_pred - z1)' * Q * (z1_pred - z1)
	l += 0.5 * (z1 - ẑ1)' * R * (z1 - ẑ1)
	return l
end

function prediction_jacobian_state!(dz, z, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance

	u = zeros(mechanism.dimensions.input)
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics_jacobian_state(dz, mechanism, z, u, w=w, idx_parameters=idx_parameters)
	get_next_state!(z1_pred, mechanism)
	return z1_pred
end

function prediction_jacobian_parameters!(dw, z, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance

	u = zeros(mechanism.dimensions.input)
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics_jacobian_parameters(dw, mechanism, z, u, w=w, idx_parameters=idx_parameters)
	get_next_state!(z1_pred, mechanism)
	return z1_pred
end

# traj loss (z1:n, w1:n)
function trajectory_loss(ẑ, z, z0, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	l = 0.0
	z_prev = deepcopy(z0)
	for i = 1:H
		l += prediction_loss(ẑ[i], z[i], z_prev, w[i], idx_parameters, mechanism; complementarity_tolerance=complementarity_tolerance)
		z_prev .= z[i]
	end
	return l
end

# traj gradient wrt z1:n w1:n
function trajectory_gradient(ẑ, z, z0, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	nz = mechanism.dimensions.state
	nw = length(idx_parameters)

	Q = I
	R = I
	dz = zeros(nz, nz)
	dw = zeros(nz, nw)
	grad = zeros(H * (nz + nw))
	z_prev = deepcopy(z0)

	for i = 1:H
		z1_pred = prediction_jacobian_state!(dz, z_prev, w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)
		z1_pred = prediction_jacobian_parameters!(dw, z_prev, w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)
		idx_zi = (i-1)*(nz+nw) .+ (1:nz)
		idx_wi = (i-1)*(nz+nw) + nz .+ (1:nw)
		idx_zi_1 = (i-2)*(nz+nw) .+ (1:nz)

		grad[idx_zi] .+= Q * (z[i] - z1_pred) + R * (z[i] - ẑ[i])
		grad[idx_wi] .+= -dw' * Q * (z[i] - z1_pred)
		if i > 1
			grad[idx_zi_1] .+= -dz' * Q * (z[i] - z1_pred)
		end
		z_prev .= z[i]
	end
	return grad
end


# traj quasi newton hessian wrt z1:n w1:n
function trajectory_hessian(ẑ, z, z0, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z)
	nz = mechanism.dimensions.state
	nw = length(idx_parameters)

	Q = I(nz)
	R = I(nz)
	dz = zeros(nz, nz)
	dw = zeros(nz, nw)
	hess = spzeros(H * (nz + nw), H * (nz + nw))
	z_prev = deepcopy(z0)

	for i = 1:H
		idx_zi = (i-1)*(nz+nw) .+ (1:nz)
		idx_wi = (i-1)*(nz+nw) + nz .+ (1:nw)
		idx_zi_1 = (i-2)*(nz+nw) .+ (1:nz)

		z1_pred = prediction_jacobian_state!(dz, z_prev, w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)
		z1_pred = prediction_jacobian_parameters!(dw, z_prev, w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)

		hess[idx_zi, idx_zi] .+= Q + R
		hess[idx_zi, idx_wi] .+= - Q * dw
		hess[idx_wi, idx_zi] .+= - dw' * Q

		hess[idx_wi, idx_wi] .+= dw' * Q * dw

		if i > 1
			hess[idx_zi, idx_zi_1] .+= - Q * dz
			hess[idx_zi_1, idx_zi] .+= - dz' * Q
			hess[idx_wi, idx_zi_1] .+= dw' * Q * dz
			hess[idx_zi_1, idx_wi] .+= dz' * Q * dw
			hess[idx_zi_1, idx_zi_1] .+= dz' * Q * dz
		end
		z_prev .= z[i]
	end
	return hess
end

@warn "TODO add tests"
#
# xrand = 0.02*ones(H0*(nz+nw))
# xrand .+= deepcopy(xtruth)
# g10 = FiniteDiff.finite_difference_gradient(x -> local_loss(x), xrand)
# g20 = local_grad(xrand)
# plot(g10)
# plot!(g20)
# plot!((g10 - g20) ./ (1e-3 .+ abs.(g10) + abs.(g20)) * 2)
# plot(g10[7:7:end])
# plot!(g20[7:7:end])
# plot!(g10[7:7:end] - g20[7:7:end])
#
#
# H10 = FiniteDiff.finite_difference_hessian(x -> local_loss(x), xrand)
# H20 = local_hess(xrand)
# plot(H10)
# plot!(H20)
# plot!(H10 - H20)
# plot(H10 - H20)
#
# plot(Gray.(abs.(H10)))
# plot(Gray.(abs.(Matrix(H20))))
# plot(Gray.(abs.(Matrix(H20 - H10))))
