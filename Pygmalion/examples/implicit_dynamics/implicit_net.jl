function pack_variables(v, θ)
	return [vcat(v...); θ]
end

function unpack_variables(vars, nv, nθ)
	H = Int((length(vars) - nθ) / nv)
	v = [vars[(i-1)*nv .+ (1:nv)] for i = 1:H]
	θ = vars[H * nv .+ (1:nθ)]
	return v, θ
end

function get_all_vars(v2, v3, θ, mechanism::Mechanism)
	# v2 = x2, [v15, ....]
	# v3 = x3, [v25, ....]
	# θ = body halfspace parameters
	body = mechanism.bodies[1]
	contact = mechanism.contacts[1]

	body_parameters = get_parameters(body)
	body_parameters = [v2[1:6]; body_parameters[7:end]] # set x2, v15
	contact_parameters = get_parameters(contact)
	contact_parameters = [contact_parameters[1]; θ; contact_parameters[end-2:end]] # set all the body halfspaces
	parameters = [body_parameters; contact_parameters]
	variables = v3[4:end]

	all_vars = [variables; parameters]
	return all_vars
end

function get_all_vars_jacobian(v2, v3, θ, mechanism::Mechanism; jacobian=:v2)
	if jacobian == :v2
		ForwardDiff.jacobian(v2 -> get_all_vars(v2, v3, θ, mechanism), v2)
	elseif jacobian == :v3
		ForwardDiff.jacobian(v3 -> get_all_vars(v2, v3, θ, mechanism), v3)
	elseif jacobian == :θ
		ForwardDiff.jacobian(θ -> get_all_vars(v2, v3, θ, mechanism), θ)
	end
end

function complete_residual(all_vars, mechanism::Mechanism)
	# dimensions
	dimensions = mechanism.solver.dimensions
	num_variables = dimensions.variables
	num_equality = dimensions.equality
	num_cone = dimensions.cone
	# indices
	indices = mechanism.solver.indices
	idx_equality = indices.equality
	idx_cone_product = indices.cone_product
	idx_duals = indices.duals
	idx_slacks = indices.slacks
	# unpack
	variables = all_vars[1:num_variables]
	parameters = all_vars[num_variables+1:end]
	slacks = variables[idx_slacks]
	duals = variables[idx_duals]


	# methods
	equality_constraint = mechanism.solver.methods.equality_constraint
	cone_constraint = mechanism.solver.cone_methods.product

	# compute residual with fast methods
	residual = zeros(num_variables + 2 * num_cone)
	@views residual_equality = residual[idx_equality]
	@views residual_cone_product = residual[idx_cone_product]
	equality_constraint(residual_equality, variables, parameters)
	cone_constraint(residual_cone_product, duals, slacks)
	# compute positivity constraints
	residual[num_equality + num_cone + 1:end] .= [min.(0, duals); min.(0, slacks)]
	return residual
end

function complete_residual_jacobian(all_vars, mechanism::Mechanism)
	# dimensions
	dimensions = mechanism.solver.dimensions
	num_variables = dimensions.variables
	num_equality = dimensions.equality
	num_cone = dimensions.cone
	num_parameters = dimensions.parameters
	# indices
	indices = mechanism.solver.indices
	idx_equality = indices.equality
	idx_cone_product = indices.cone_product
	idx_duals = indices.duals
	idx_slacks = indices.slacks
	# unpack
	variables = all_vars[1:num_variables]
	parameters = all_vars[num_variables+1:end]
	slacks = variables[idx_slacks]
	duals = variables[idx_duals]

	# evaluate jacobian
	problem = mechanism.solver.problem
	data = mechanism.solver.data
	methods = mechanism.solver.methods
	cone_methods = mechanism.solver.cone_methods
	solution = mechanism.solver.solution
	solution.all .= variables
	Mehrotra.evaluate!(problem,
	        methods,
	        cone_methods,
	        solution,
	        parameters;
	        equality_jacobian_variables=true,
	        equality_jacobian_parameters=true,
			equality_jacobian_keywords=[:all],
	        cone_jacobian=true,
	        )
	# fill jacobian
	Mehrotra.residual!(data, problem, indices;
	        jacobian_variables=true,
	        jacobian_parameters=true,
			)
	jacobian = zeros(num_variables + 2 * num_cone, num_variables + num_parameters)
	idx_variables = 1:num_variables
	idx_parameters = num_variables .+ (1:num_parameters)
	jacobian[idx_variables, idx_variables] .= data.jacobian_variables_dense
	jacobian[idx_variables, idx_parameters] .= data.jacobian_parameters
	jacobian[num_variables .+ (1:num_cone), idx_duals] .= (duals .<= 0.0)
	jacobian[num_variables + num_cone .+ (1:num_cone), idx_slacks] .= (slacks .<= 0.0)
	return jacobian
end

function residual_objective(residual, mechanism::Mechanism;
		Q_equality=I,
		Q_cone_product=I,
		Q_duals=I,
		Q_slacks=I,
		ρ=1e-3)
	# TODO generalize to second order cone

	indices = mechanism.solver.indices
	idx_equality = indices.equality
	idx_cone_product = indices.cone_product

	dimensions = mechanism.solver.dimensions
	num_equality = dimensions.equality
	num_cone = dimensions.cone

	equality = residual[idx_equality]
	cone_product = residual[idx_cone_product] .- ρ
	duals_positivity = residual[num_equality + num_cone .+ (1:num_cone)]
	slacks_positivity = residual[num_equality + 2 * num_cone .+ (1:num_cone)]

	r = 0.0
	r += 0.5 * equality' * Q_equality * equality
	r += 0.5 * cone_product' * Q_cone_product * cone_product
	r += 0.5 * duals_positivity' * Q_duals * duals_positivity
	r += 0.5 * slacks_positivity' * Q_slacks * slacks_positivity
	r_equality = 0.5 * equality' * Q_equality * equality
	r_cone = 0.5 * cone_product' * Q_cone_product * cone_product
	r_duals = 0.5 * duals_positivity' * Q_duals * duals_positivity
	r_slacks = 0.5 * slacks_positivity' * Q_slacks * slacks_positivity
	# if eltype(r_duals) <: Float64
	# 	@show r_equality
	# 	@show r_cone
	# 	@show r_duals
	# 	@show r_slacks
	# end
	return r
end

function residual_objective_jacobian(residual, mechanism::Mechanism; kwargs...)
	grad = ForwardDiff.gradient(residual -> residual_objective(residual, mechanism; kwargs...), residual)
	return grad
end

function step_objective(v2, v3, θ, mechanism::Mechanism;
		Q_integrator=I,
		kwargs...)

	all_vars = get_all_vars(v2, v3, θ, mechanism)
	residual = complete_residual(all_vars, mechanism)
	# @show "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
	objective = residual_objective(residual, mechanism; kwargs...)

	timestep = mechanism.bodies[1].timestep
	r_integrator = integrator_objective(v2, v3, timestep, Q_integrator=Q_integrator)
	# @show r_integrator
	return objective + integrator_objective(v2, v3, timestep, Q_integrator=Q_integrator)
end

function step_objective_jacobian(v2, v3, θ, mechanism::Mechanism;
		Q_integrator=I,
		kwargs...)

	dvars_dv2 = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:v2)
	dvars_dv3 = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:v3)
	dvars_dθ = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:θ)

	all_vars = get_all_vars(v2, v3, θ, mechanism)
	residual = complete_residual(all_vars, mechanism)
	dr_dvars = complete_residual_jacobian(all_vars, mechanism)

	dj_dr = residual_objective_jacobian(residual, mechanism; kwargs...)
	dv2 = dvars_dv2' * dr_dvars' * dj_dr
	dv3 = dvars_dv3' * dr_dvars' * dj_dr
	dθ = dvars_dθ' * dr_dvars' * dj_dr


	timestep = mechanism.bodies[1].timestep
	dv2 .+= ForwardDiff.gradient(v2 ->
		integrator_objective(v2, v3, timestep, Q_integrator=Q_integrator), v2)
	dv3 .+= ForwardDiff.gradient(v3 ->
		integrator_objective(v2, v3, timestep, Q_integrator=Q_integrator), v3)

	return dv2, dv3, dθ
end

function integrator_objective(v2, v3, timestep; Q_integrator=I)
	p2 = v2[1:3]
	p3 = v3[1:3]
	v25 = v3[4:6]
	return 0.5 * (p2 + timestep .* v25 - p3)' * Q_integrator * (p2 + timestep .* v25 - p3)
end

function traj_objective(v0, v, θ, mechanism::Mechanism; kwargs...)
	H = length(v)
	objective = step_objective(v0, v[1], θ, mechanism; kwargs...)
	for i = 2:H
		objective += step_objective(v[i-1], v[i], θ, mechanism; kwargs...)
	end
	return objective
end

function traj_objective_jacobian(v0, v, θ, mechanism::Mechanism; kwargs...)
	H = length(v)
	nv = length(v0)
	nθ = length(θ)

	dv = [zeros(nv) for i = 1:H]
	dθ = zeros(nθ)

	dv2i, dv3i, dθi = step_objective_jacobian(v0, v[1], θ, mechanism; kwargs...)
	dv[1] .+= dv3i
	dθ .+= dθi

	for i = 2:H
		dv2i, dv3i, dθi = step_objective_jacobian(v[i-1], v[i], θ, mechanism; kwargs...)
		dv[i-1] .+= dv2i
		dv[i] .+= dv3i
		dθ .+= dθi
	end

	return dv, dθ
end

function traj_objective_hessian(v0, v, θ, mechanism::Mechanism;
		Q_equality=I,
		Q_cone_product=I,
		Q_duals=I,
		Q_slacks=I,
		Q_integrator=I,
		ρ=1e-3)

	# dimensions
	dimensions = mechanism.solver.dimensions
	num_equality = dimensions.equality
	num_cone = dimensions.cone
	num_cone = dimensions.cone
	timestep = mechanism.bodies[1].timestep[1]

	H = length(v)
	nv = length(v0)
	nθ = length(θ)
	nr = num_equality + 3 * num_cone

	dr_dv2 = [zeros(nr, nv) for i = 1:H-1]
	dr_dv3 = [zeros(nr, nv) for i = 1:H]
	dr_dθ = [zeros(nr, nθ) for i = 1:H]
	hess = spzeros(nv * H + nθ, nv * H + nθ)

	for i = 1:H
		v_prev = (i == 1) ? v0 : v[i-1]
		dvars_dv2 = get_all_vars_jacobian(v_prev, v[i], θ, mechanism, jacobian=:v2)
		dvars_dv3 = get_all_vars_jacobian(v_prev, v[i], θ, mechanism, jacobian=:v3)
		dvars_dθ = get_all_vars_jacobian(v_prev, v[i], θ, mechanism, jacobian=:θ)
		all_vars = get_all_vars(v_prev, v[i], θ, mechanism)
		residual = complete_residual(all_vars, mechanism)
		dr_dvars = complete_residual_jacobian(all_vars, mechanism)
		(i > 1) && (dr_dv2[i-1] = dr_dvars * dvars_dv2)
		dr_dv3[i] = dr_dvars * dvars_dv3
		dr_dθ[i] = dr_dvars * dvars_dθ
	end

	Q_residual = Diagonal([
		diag(Q_equality(num_equality));
		diag(Q_cone_product(num_cone));
		diag(Q_duals(num_cone));
		diag(Q_slacks(num_cone));
		])

	ind_θ = nv * H .+ (1:nθ)
	for i = 1:H
		ind_v2 = (i-2)*nv .+ (1:nv)
		ind_v3 = (i-1)*nv .+ (1:nv)
		hess[ind_θ, ind_θ]   .+= dr_dθ[i]'  * Q_residual * dr_dθ[i]
		hess[ind_v3, ind_θ]  .+= dr_dv3[i]' * Q_residual * dr_dθ[i]
		hess[ind_θ, ind_v3]  .+= dr_dθ[i]'  * Q_residual * dr_dv3[i]
		hess[ind_v3, ind_v3] .+= dr_dv3[i]' * Q_residual * dr_dv3[i]
		if i > 1
			hess[ind_θ, ind_v2] .+= dr_dθ[i]' * Q_residual * dr_dv2[i-1]
			hess[ind_v2, ind_θ] .+= dr_dv2[i-1]' * Q_residual * dr_dθ[i]
			hess[ind_v2, ind_v3] .+= dr_dv2[i-1]' * Q_residual * dr_dv3[i]
			hess[ind_v3, ind_v2] .+= dr_dv3[i]' * Q_residual * dr_dv2[i-1]
			hess[ind_v2, ind_v2] .+= dr_dv2[i-1]' * Q_residual * dr_dv2[i-1]
		end
		# integrator
		ind_v23 = [ind_v2; ind_v3]
		v2 = (i == 1) ? v0 : v[i-1]
		v3 = v[i]
		if i > 1
			hess[ind_v23, ind_v23] .+= ForwardDiff.hessian(v23 ->
				integrator_objective(v23[1:nv], v23[nv .+ (1:nv)], timestep, Q_integrator=Q_integrator),
				[v2; v3])
		else
			hess[ind_v3, ind_v3] .+= ForwardDiff.hessian(v3 ->
				integrator_objective(v2, v3, timestep, Q_integrator=Q_integrator),
				v3)
		end
	end
	return hess
end





function integrator_constraint(v2, v3, timestep)
	p2 = v2[1:3]
	p3 = v3[1:3]
	v25 = v3[4:6]
	return p2 + timestep .* v25 - p3
end

function step_constraint(v2, v3, θ, mechanism::Mechanism;
		ρ=1e-3)

	all_vars = get_all_vars(v2, v3, θ, mechanism)
	residual = complete_residual(all_vars, mechanism)

	timestep = mechanism.bodies[1].timestep
	integrator = integrator_constraint(v2, v3, timestep)
	return [integrator; residual]
end

function step_constraint_jacobian(v2, v3, θ, mechanism::Mechanism;
		ρ=1e-3)

	dvars_dv2 = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:v2)
	dvars_dv3 = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:v3)
	dvars_dθ = get_all_vars_jacobian(v2, v3, θ, mechanism, jacobian=:θ)

	all_vars = get_all_vars(v2, v3, θ, mechanism)
	residual = complete_residual(all_vars, mechanism)
	dr_dvars = complete_residual_jacobian(all_vars, mechanism)

	dr_dv2 = dr_dvars * dvars_dv2
	dr_dv3 = dr_dvars * dvars_dv3
	dr_dθ = dr_dvars * dvars_dθ


	timestep = mechanism.bodies[1].timestep
	di_dv2 = ForwardDiff.jacobian(v2 ->
		integrator_constraint(v2, v3, timestep), v2)
	di_dv3 = ForwardDiff.jacobian(v3 ->
		integrator_constraint(v2, v3, timestep), v3)
	di_dθ = ForwardDiff.jacobian(θ ->
		integrator_constraint(v2, v3, timestep), θ)

	dv2 = [di_dv2; dr_dv2]
	dv3 = [di_dv3; dr_dv3]
	dθ = [di_dθ; dr_dθ]
	return dv2, dv3, dθ
end



function traj_constraints(v0, v, θ, mechanism::Mechanism; kwargs...)
	H = length(v)
	objective = step_objective(v0, v[1], θ, mechanism; kwargs...)
	for i = 2:H
		objective += step_objective(v[i-1], v[i], θ, mechanism; kwargs...)
	end
	return objective
end

function traj_objective_jacobian(v0, v, θ, mechanism::Mechanism; kwargs...)
	H = length(v)
	nv = length(v0)
	nθ = length(θ)

	dv = [zeros(nv) for i = 1:H]
	dθ = zeros(nθ)

	dv2i, dv3i, dθi = step_objective_jacobian(v0, v[1], θ, mechanism; kwargs...)
	dv[1] .+= dv3i
	dθ .+= dθi

	for i = 2:H
		dv2i, dv3i, dθi = step_objective_jacobian(v[i-1], v[i], θ, mechanism; kwargs...)
		dv[i-1] .+= dv2i
		dv[i] .+= dv3i
		dθ .+= dθi
	end

	return dv, dθ
end
