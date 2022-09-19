include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("contact_net.jl")

vis = Visualizer()
# render(vis)
open(vis)
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

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0,
	name=:poly0, color=RGBA(1,1,1,1.0))

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
Mehrotra.solve!(mech.solver)


################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.50,-0.00]
vp15 = [-0,0,-9.0]
z0 = [xp2; vp15]

H0 = 10
u0 = [0.0, 0.0, 0.0]
ctrl = open_loop_controller([u0])

storage = simulate!(mech, z0, H0, controller=ctrl)
vis, anim = visualize!(vis, mech, storage)
plot(hcat([x[1] for x in storage.x]...)')
plot(hcat([x for x in storage.z]...)')
plot(hcat([x[mech.solver.indices.optimality] for x in storage.variables]...)')
plot(hcat([x[mech.solver.indices.slackness] for x in storage.variables]...)')
plot(hcat([x[mech.solver.indices.cone_product] for x in storage.variables]...)')

################################################################################
# camera parameters
################################################################################
Mehrotra.solve!(mech.solver)
primals = deepcopy(Vector(mech.solver.solution.primals))
duals = deepcopy(Vector(mech.solver.solution.duals))
slacks = deepcopy(Vector(mech.solver.solution.slacks))
parameters = deepcopy(mech.solver.parameters)

cost = implicit_residual(primals, duals, parameters, mech)
FiniteDiff.finite_difference_gradient(primals ->
	implicit_residual(primals, duals, parameters, mech), primals)
FiniteDiff.finite_difference_gradient(duals ->
	implicit_residual(primals, duals, parameters, mech), duals)
FiniteDiff.finite_difference_gradient(parameters ->
	implicit_residual(primals, duals, parameters, mech), parameters)




# nx = 6 + 6 + 3 + 9 + 1

H = H0 - 1
nx = 6 + 3 + 9 + 1
states = [deepcopy(storage.z[i]) for i=1:H+1]
contact_primals = [deepcopy(storage.variables[i][mech.contacts[1].index.primals]) for i=1:H]
duals = [deepcopy(storage.variables[i][mech.solver.indices.duals]) for i=1:H]
learnable_parameters = [deepcopy(mech.contacts[1].friction_coefficient) for i=1:H]

x = vcat([
	[
	states[i+1];
	contact_primals[i];
	duals[i];
 	learnable_parameters[i]
	] for i=1:H]...)
measured_poses = [states[i+1][1:3] for i=1:H]
initial_state = states[1]
total_cost(x, initial_state, measured_poses, mech)

local_cost(x) = total_cost(x, initial_state, measured_poses, mech)
local_grad(x) = FiniteDiff.finite_difference_gradient(x -> total_cost(x, initial_state, measured_poses, mech), x)
function total_hess(x, initial_state, measured_poses, mech)
	H = length(measured_poses)
	nx = Int(length(x) / H)
	n = nx * H
	hess = spzeros(n, n)
	#
	# off = 0
	# for i = 1:H
	# 	ind = off .+ (1:nx)
	# 	off += nx
	# 	function local_cost(xi::Vector{T}) where T
	# 		xl = zeros(T, n)
	# 		xl .= x
	# 		xl[ind] .= xi
	# 		total_cost(xl, initial_state, measured_poses, mech)
	# 	end
	# 	hess[ind, ind] .= FiniteDiff.finite_difference_hessian(xi -> local_cost(xi), x[ind])
	# end
	# return hess
	I(n)
end
local_hess(x) = total_hess(x, initial_state, measured_poses, mech)

local_cost(x)
local_grad(x)
hess = local_hess(x)


function eq_fct(x)
 	friction_coefficients = x[nx:nx:end]
	return friction_coefficients[1:end-1] .- friction_coefficients[2:end]
end

model = Nonconvex.Model()

n = nx * H
lb = -1e3
ub = +1e3
xinit = deepcopy(x)

addvar!(model, lb*ones(n), ub*ones(n), init=xinit, integer=falses(n))
add_eq_constraint!(model, eq_fct)

alg = AugLag()
options = AugLagOptions()

F = Nonconvex.CustomHessianFunction(local_cost, local_grad, local_hess)
set_objective!(model, F)

result = Nonconvex.optimize(model, alg, xinit, options=options)
result.minimizer



# alg = IpoptAlg()
# verbose = false
# verbose = true
# max_cpu_time = 3.0
# print_level = verbose ? 5 : 0
# options = IpoptOptions(print_level=print_level, max_cpu_time=max_cpu_time)

# i0 = 19
# state = deepcopy(storage.z[i0])
# next_state = deepcopy(storage.z[i0+1])
# contact_primals = deepcopy(storage.variables[i0][mech.contacts[1].index.primals])
# duals = deepcopy(storage.variables[i0][mech.solver.indices.duals])
# learnable_parameters = deepcopy(mech.contacts[1].friction_coefficient)
# learnable_parameters = [1.2]
# stage_cost(state, next_state, contact_primals, duals, learnable_parameters, mech)
#
# function loss(i0, friction_coefficient)
# 	state = deepcopy(storage.z[i0])
# 	next_state = deepcopy(storage.z[i0+1])
# 	contact_primals = deepcopy(storage.variables[i0][mech.contacts[1].index.primals])
# 	duals = deepcopy(storage.variables[i0][mech.solver.indices.duals])
# 	learnable_parameters = [friction_coefficient]
# 	return dynamics_cost(state, next_state, contact_primals, duals, learnable_parameters, mech)
# end
#
# plt = plot()
# for i = 1:H0-1
# 	plot!(plt, [loss(i, f) for f in range(-1, 2, length=1000)], linewidth=i)
# 	display(plt)
# 	sleep(0.1)
# end
# display(plt)
