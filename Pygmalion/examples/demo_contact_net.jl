include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("implicit_net.jl")

vis = Visualizer()
render(vis)
# open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))


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
op0 = [0.0, 0.0]
Af = [0.0  +1.0]
bf = [0.0]
of = [0.0]

Ap = [Ap0, Af]
bp = [bp0, bf]
op = [op0, of]
θ_p, polytope_dimensions_p = pack_halfspaces(Ap, bp, op)

# build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0,
	# name=:poly0, color=RGBA(1,1,1,1.0))

################################################################################
# inertial parameters
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.1,
	# method_type=:finite_difference,
    method_type=:symbolic,
	A=Ap0,
	b=bp0,
    options=Mehrotra.Options(
        verbose=false,
		residual_tolerance=1e-4,
        complementarity_tolerance=1e-3,
		# compressed_search_direction=true,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );
Mehrotra.solve!(mech.solver)

parameter_dimension(mech.bodies[1])
parameter_dimension(mech.contacts[1])

################################################################################
# test simulation
################################################################################
# xp2 = [+0.0,1.50,-0.00]
xp2 = [+0.0,0.5,-0.00]
# vp15 = [-0,0,-9.0]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]

H0 = 10

storage = simulate!(mech, z0, H0+1)
vis, anim = visualize!(vis, mech, storage)

################################################################################
# camera parameters
################################################################################
# true data
num_variables = mech.solver.dimensions.variables
θ_truth = deepcopy([vec(Ap0); bp0])
mech.solver.parameters[15:26] .= θ_truth
update_nodes!(mech)
v0_truth = deepcopy([xp2; vp15; zeros(num_variables - 3)])
v_truth = deepcopy([[storage.x[i+1][1]; storage.variables[i]] for i = 1:H0])
x_truth = deepcopy([storage.x[i+1][1] for i = 1:H0])
vars_truth = pack_variables(v_truth, θ_truth)
nv = length(v0_truth)
nθ = length(θ_truth)

# # true data
# θ_plaus = deepcopy([vec(Ap0); bp0 .+ 0.1])
# mech.solver.parameters[15:26] .= θ_plaus
# update_nodes!(mech)
# storage_plaus = simulate!(mech, z0, H0+1)
# v0_plaus = deepcopy([xp2; vp15; zeros(num_variables - 3)])
# v_plaus = deepcopy([[storage_plaus.x[i+1][1]; storage_plaus.variables[i]] for i = 1:H0])
# vars_plaus = pack_variables(v_plaus, θ_plaus)
# # vars_plaus = pack_variables(v_truth, θ_plaus)
# # vis, anim = visualize!(vis, mech, storage_plaus)


# fake data
# θ_fake = θ_truth .+ 0.4 * rand(12)
θ_fake = deepcopy(θ_truth .+ 0.1 * [zeros(8); ones(4)])
v_fake = [[deepcopy(v[1:6]); 0.0*ones(21)] for v in v_truth]
for i = 1:H0
	z0 = (i == 1) ? v0_truth[1:6] : v_fake[i-1][1:6]
	mech.solver.parameters[15:26] .= θ_fake
	update_nodes!(mech)
	storage_i = simulate!(mech, z0, 2)
	v_fake[i][7:end] .= storage_i.variables[1][4:end]
end
vars_fake = pack_variables(v_fake, θ_fake)

mech.solver.parameters[15:26] .= θ_truth
update_nodes!(mech)

# weights
Q_equality = 1.0 * 1.0 * I
Q_cone_product = 1.0 * 1.0 * I
Q_duals = 1.0 * 1.0 * I
Q_slacks = 1.0 * 1.0 * I
Q_integrator = 1.0 * 1.0 * I
Q_tracking = 1.0 * 110.0 * I

function tracking_objective(vars, x; Q_tracking=I)
	v, θ = unpack_variables(vars, nv, nθ)
	H = length(v)
	objective = 0.0
	for i = 1:H
		objective += 0.5 * (v[i][1:3] - x[i])' * Q_tracking * (v[i][1:3] - x[i])
	end
	return objective
end

function tracking_jacobian(vars, x; Q_tracking=I)
	ForwardDiff.gradient(vars -> tracking_objective(vars, x; Q_tracking=Q_tracking), vars)
end

function tracking_hessian(vars, x; Q_tracking=I)
	d = vcat([[Q_tracking(3); zeros(nv-3)] for i = 1:H]...)
	return Diagonal([d; ones(nθ)])
end

local_loss(vars) = traj_objective(v0_truth, unpack_variables(vars, nv, nθ)..., mech,
		Q_equality=Q_equality,
		Q_cone_product=Q_cone_product,
		Q_duals=Q_duals,
		Q_slacks=Q_slacks,
		Q_integrator=Q_integrator,
		) + tracking_objective(vars, x_truth, Q_tracking=Q_tracking)

local_grad(vars) = pack_variables(traj_objective_jacobian(v0_truth, unpack_variables(vars, nv, nθ)..., mech,
		Q_equality=Q_equality,
		Q_cone_product=Q_cone_product,
		Q_duals=Q_duals,
		Q_slacks=Q_slacks,
		Q_integrator=Q_integrator,
		)...) + tracking_jacobian(vars, x_truth, Q_tracking=Q_tracking)

local_hess(vars) = traj_objective_hessian(v0_truth, v_truth, θ_truth, mech,
		Q_equality=Q_equality,
		Q_cone_product=Q_cone_product,
		Q_duals=Q_duals,
		Q_slacks=Q_slacks,
		Q_integrator=Q_integrator,
		) + tracking_hessian(vars, x_truth, Q_tracking=Q_tracking)


# local_loss(vars_plaus)
local_loss(vars_plaus)
local_loss(vars_truth)
local_loss(vars_fake)

# projection
function local_projection(vars)
	θ = vars[end-nθ+1:end]
	θp = projection([θ; zeros(2)], [4],
		Alims=[-1.00, +1.00],
		blims=[+0.02, +0.60],
		olims=[-3.00, +3.00],
		)
	vars[end-nθ+1:end] .= θp[1:end-2]
	return vars
end

################################################################################
# adam solve
################################################################################
adam_opt = Adam(vars_fake, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 2e-4
max_iterations = 5000
vars_sol0, vars_iter0 = adam_solve!(adam_opt, projection=local_projection,
 	max_iterations=max_iterations, l_tolerance=1e-116)

v_sol0, θ_sol0 = unpack_variables(vars_sol0, nv, nθ)
θ_iter0 = [[unpack_variables(v, nv, nθ)[2]; zeros(2)] for v in vars_iter0]

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0,
	name=:poly0, color=RGBA(1,1,1,1))
visualize_polytope_iterates!(vis, θ_iter0[1:Int(floor(max_iterations/100)):end], [4], color=RGBA(0,1,1,0.5))
set_floor!(vis, origin=[0,0,-0.5], x=0.02, y=10)

# plot(hcat([v[1:3] for v in v_sol0]...)')





################################################################################
# newton solve
################################################################################
vars_sol0, vars_iter0 = newton_solver!(vars_fake,
		local_loss,
		local_grad,
		local_hess,
		local_projection,
		# x->x,
		x->x;
        max_iterations=250,
        reg_min=1e-2,
        reg_max=1e-0,
        reg_step=1.1,
        line_search_iterations=10,
        residual_tolerance=1e-119,
		)

v_sol0, θ_sol0 = unpack_variables(vars_sol0, nv, nθ)
θ_iter0 = [[unpack_variables(v, nv, nθ)[2]; zeros(2)] for v in vars_iter0]

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0,
	name=:poly0, color=RGBA(1,1,1,1))
visualize_polytope_iterates!(vis, θ_iter0, [4], color=RGBA(0,1,1,0.5))
set_floor!(vis, origin=[0,0,-0.5], x=0.02, y=10)
