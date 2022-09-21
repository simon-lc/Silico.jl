

################################################################################
# ground truth
################################################################################

Ap0 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(Ap0)
bp0 = 0.5*[
	+1,
	+1,
	+1,
	+1,
	]

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.1,
	method_type=:finite_difference,
    # method_type=:symbolic,
	A=Ap0,
	b=bp0,
    options=Mehrotra.Options(
        verbose=false,
		residual_tolerance=1e-7,
        complementarity_tolerance=1e-6,
		# compressed_search_direction=true,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );


xp2 = [+0.0,1.50,-0.00]
vp15 = [-0,0,-9.0]
z0 = [xp2; vp15]
θ = rand(12)




num_variables = mech.solver.dimensions.variables
θ_truth = deepcopy([vec(Ap0); bp0])
v0_truth = deepcopy([xp2; vp15; zeros(num_variables - 3)])
v_truth = deepcopy([[storage.x[i+1][1]; storage.variables[i]] for i = 1:H0])

θ_fake = deepcopy(θ_truth .+ 0.1rand(12))

plot(hcat(v_truth...)', legend=false)

step_objective(v0_truth, v_truth[1], θ_truth, mech)
traj_objective(v0_truth, v_truth, θ_truth, mech)





H = 10
nv = length(v0_truth)
nθ = length(θ_truth)

v20 = rand(nv)
v30 = rand(nv)
θ0 = rand(nθ)
v0 = rand(nv)
v = [rand(nv) for i = 1:H]

dvars_dv2 = get_all_vars_jacobian(v20, v30, θ0, mech, jacobian=:v2)
dvars_dv3 = get_all_vars_jacobian(v20, v30, θ0, mech, jacobian=:v3)
dvars_dθ = get_all_vars_jacobian(v20, v30, θ0, mech, jacobian=:θ)
dvars_dv20 = FiniteDiff.finite_difference_jacobian(
	v20 -> get_all_vars(v20, v30, θ0, mech), v20)
dvars_dv30 = FiniteDiff.finite_difference_jacobian(
	v30 -> get_all_vars(v20, v30, θ0, mech), v30)
dvars_dθ0 = FiniteDiff.finite_difference_jacobian(
	θ0 -> get_all_vars(v20, v30, θ0, mech), θ0)
all_vars0 = get_all_vars(v20, v30, θ0, mech)

norm(dvars_dv2)
norm(dvars_dv3)
norm(dvars_dθ)
norm(dvars_dv2 - dvars_dv20)
norm(dvars_dv3 - dvars_dv30)
norm(dvars_dθ - dvars_dθ0)



dr_dvars = complete_residual_jacobian(all_vars0, mech)
dr_dvars0 = FiniteDiff.finite_difference_jacobian(all_vars0 -> complete_residual(all_vars0, mech), all_vars0)
residual0 = complete_residual(all_vars0, mech)

norm(dr_dvars)
norm(dr_dvars - dr_dvars0)


dj_dr = residual_objective_jacobian(residual0, mech)
dj_dr0 = FiniteDiff.finite_difference_gradient(residual0 -> residual_objective(residual0, mech), residual0)

norm(dj_dr)
norm(dj_dr - dj_dr0)


dv2, dv3, dθ = step_objective_jacobian(v20, v30, θ0, mech)
dv20 = FiniteDiff.finite_difference_gradient(
	v20 -> step_objective(v20, v30, θ0, mech), v20)
dv30 = FiniteDiff.finite_difference_gradient(
	v30 -> step_objective(v20, v30, θ0, mech), v30)
dθ0 = FiniteDiff.finite_difference_gradient(
	θ0 -> step_objective(v20, v30, θ0, mech), θ0)

norm(dv2)
norm(dv3)
norm(dθ)
norm(dv2 - dv20)
norm(dv3 - dv30)
norm(dθ - dθ0)



traj_objective(v0, v, θ, mech)
objective = step_objective(v0, v[1], θ, mech)
dv, dθ = traj_objective_jacobian(v0, v, θ, mech)
dθ0 = FiniteDiff.finite_difference_gradient(
	θ -> traj_objective(v0, v, θ, mech), θ)
dv0 = [FiniteDiff.finite_difference_gradient(
	vi -> traj_objective(v0, deepcopy([v[1:i-1]..., vi, v[i+1:end]...]), θ, mech), v[i])
	for i = 1:H]

sum(norm.(dv .- dv0))
norm(dθ - dθ0)




dv2, dv3, dθ = step_constraint_jacobian(v20, v30, θ0, mech)
dv20 = FiniteDiff.finite_difference_jacobian(
	v20 -> step_constraint(v20, v30, θ0, mech), v20)
dv30 = FiniteDiff.finite_difference_jacobian(
	v30 -> step_constraint(v20, v30, θ0, mech), v30)
dθ0 = FiniteDiff.finite_difference_jacobian(
	θ0 -> step_constraint(v20, v30, θ0, mech), θ0)

norm(dv2)
norm(dv3)
norm(dθ)
norm(dv2 - dv20)
norm(dv3 - dv30)
norm(dθ - dθ0)





@elapsed hess = traj_objective_hessian(v0_truth, v_truth, θ_truth, mech, Q_integrator=I)
hess0 = FiniteDiff.finite_difference_jacobian(
	vars -> pack_variables(traj_objective_jacobian(v0_truth, unpack_variables(vars, nv, nθ)..., mech, Q_integrator=I)...),
	vars_truth)

Matrix(hess)
plot(Gray.(abs.(Matrix(hess))))
plot(Gray.(abs.(Matrix(hess0))))
plot(Gray.(abs.(Matrix(hess - hess0))))
norm(hess - hess0)
norm(hess)
norm(hess0)
