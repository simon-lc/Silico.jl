function prediction_loss(z1, z0, w1, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10
	mechanism.solver.options.complementarity_tolerance = complementarity_tolerance

	u0 = zeros(mechanism.dimensions.input)
	z1_pred = zeros(mechanism.dimensions.state)

	DojoLight.dynamics(z1_pred, mechanism, z0, u0, w=w1, idx_parameters=idx_parameters)

	Q = I
	l = 0.0
	l += 0.5 * (z1_pred - z1)' * Q * (z1_pred - z1)
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
function trajectory_loss(z, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z) - 1
	l = 0.0
	for i = 1:H
		l += prediction_loss(z[i+1], z[i], w[i], idx_parameters, mechanism; complementarity_tolerance=complementarity_tolerance)
	end
	return l
end

# traj gradient wrt w
function trajectory_gradient(z, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z) - 1
	nz = mechanism.dimensions.state
	nw = length(idx_parameters)

	Q = I
	dw = zeros(nz, nw)
	grad = zeros(nw)

	for i = 1:H
		z1_pred = prediction_jacobian_parameters!(dw, z[i], w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)
		grad .+= -dw' * Q * (z[i+1] - z1_pred)
	end
	return grad
end


# traj quasi newton hessian wrt w
function trajectory_hessian(z, w, idx_parameters, mechanism; complementarity_tolerance=1e-3)
	H = length(z) - 1
	nz = mechanism.dimensions.state
	nw = length(idx_parameters)

	Q = I(nz)
	dw = zeros(nz, nw)
	hess = zeros(nw, nw)

	for i = 1:H
		z1_pred = prediction_jacobian_parameters!(dw, z[i], w[i], idx_parameters,
			mechanism; complementarity_tolerance=complementarity_tolerance)
		hess .+= dw' * Q * dw
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
