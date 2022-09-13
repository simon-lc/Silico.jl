function implicit_residual(primals, duals, parameters, mechanism::Mechanism)
	num_slacks = mechanism.solver.dimensions.slacks
	indices = mechanism.solver.indices
	slacks = zeros(num_slacks)

	# fix slacks to zero out slackness constraints
	residual = mechanism_residual(primals, duals, slacks, parameters,
		mechanism.bodies,
		mechanism.contacts)
	slacks = -residual[indices.slackness]
	cost = 0.0
	cost += sum(min.(0, slacks).^2)
	# slacks .= max.(0, slacks) # do we need this??????????????????????????????????????????????

	# compute residual
	residual = mechanism_residual(primals, duals, slacks, parameters,
		mechanism.bodies,
		mechanism.contacts)
	# compute cone product
	cone_prod = Mehrotra.cone_product(duals, slacks, indices.cone_nonnegative, indices.cone_second_order)

	cost += sum(residual[indices.optimality].^2)
	cost += sum(cone_prod.^2)
	cost += sum(min.(0, duals).^2)
	return cost
end

function dynamics_cost(state, next_state, contact_primals, duals, learnable_parameters, mechanism::Mechanism)
	# learnable_parameters = [friction_coefficient]

	parameters = mechanism.solver.parameters
	p2 = state[1:3]
	v15 = state[4:6]
	p3 = next_state[1:3]
	v25 = next_state[4:6]

	# set body parameters
	body = mechanism.bodies[1]
	body.pose .= p2
	body.velocity .= v15
	parameters[body.index.parameters] .= get_parameters(body)

	# set contact parameters
	contact = mechanism.contacts[1]
	contact.friction_coefficient .= learnable_parameters[1:1]
	parameters[contact.index.parameters] .= get_parameters(contact)

	primals = [v25; contact_primals]
	integrator_cost = sum((p2 + body.timestep[1] * v25 - p3).^2)
	cost = 1.0*implicit_residual(primals, duals, parameters, mechanism) + 1.0*integrator_cost
	return cost
end

function measurement_cost(state, measured_pose)
	pose = state[1:3]
	return sum((pose - measured_pose).^2)
end

function stage_cost(x, state, measured_pose, mechanism)
	off = 0
	next_state = x[off .+ (1:6)]; off += 6
	contact_primals = x[off .+ (1:3)]; off += 3
	duals = x[off .+ (1:9)]; off += 9
	learnable_parameters = x[off .+ (1:1)]; off += 1

	cost = 1.0*dynamics_cost(state, next_state, contact_primals, duals, learnable_parameters, mechanism) +
		1.0*measurement_cost(next_state, measured_pose)
	return cost
end

function total_cost(x, initial_state, measured_poses, mechanism)
	cost = 0.0
	H = length(measured_poses)
	for i = 1:H
		xi = x[nx*(i-1) .+ (1:nx)]
		cost += stage_cost(xi, initial_state, measured_poses[i], mechanism)
		initial_state .= xi[1:6]
	end
	return cost
end
