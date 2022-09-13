include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("prediction.jl")

################################################################################
# visualize
################################################################################
vis = Visualizer()
# render(vis)
open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))


################################################################################
# polytope parameters
################################################################################
A0 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
normalize_A!(A0)
b0 = 0.5*[
	+1,
	+1,
	+1,
	+1,
	]
o0 = [0.0, +0.0]
Af = [0.0  +1.0]
bf = [0.0]
of = [0.0]
build_2d_polytope!(vis[:polytope], A0, b0 + A0 * o0,
	name=:poly0, color=RGBA(1,1,1,1.0))


################################################################################
# inertial parameters
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
μ0 = [0.9]
complementarity_tolerance = 1e-3

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=μ0[1],
	method_type=:symbolic,
	A=A0,
	b=b0,
    options=Mehrotra.Options(
        verbose=false,
		residual_tolerance=complementarity_tolerance/10,
        complementarity_tolerance=complementarity_tolerance,
		compressed_search_direction=true,
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
xp2 = [+0.0,1.00,-0.00]
vp15 = [6,0.0,+2.0]
z0 = [xp2; vp15]

H0 = 20
u0 = [0.0, 0.0, 0.0]
ctrl = open_loop_controller([u0])

storage = simulate!(mech, z0, H0+1, controller=ctrl)
vis, anim = visualize!(vis, mech, storage)
# plot(hcat([x[1] for x in storage.x]...)')
# plot(hcat([x for x in storage.z]...)')
# plot(hcat([x[mech.solver.indices.optimality] for x in storage.variables]...)')
# plot(hcat([x[mech.solver.indices.slackness] for x in storage.variables]...)')
# plot(hcat([x[mech.solver.indices.cone_product] for x in storage.variables]...)')


################################################################################
# dynamics
################################################################################
idx_parameters = [14]
num_state = mech.dimensions.body_state
num_parameters = mech.solver.dimensions.parameters
num_learnable_parameters = length(idx_parameters)
nz = num_state
nw = num_learnable_parameters

u1 = [0, 0, 0.0]
x1 = [0, 2, 0.0]
v1 = [0, 0, 0.0]
z1 = [x1; v1]

z2 = zeros(num_state)
dz2 = zeros(num_state, num_state)
dw2 = zeros(num_state, num_learnable_parameters)
DojoLight.dynamics(z2, mech, z1, u1, w=μ0, idx_parameters=idx_parameters)
DojoLight.dynamics_jacobian_state(dz2, mech, z1, u1, w=μ0, idx_parameters=idx_parameters)
DojoLight.dynamics_jacobian_parameters(dw2, mech, z1, u1, w=μ0, idx_parameters=idx_parameters)


################################################################################
# prediction
################################################################################
# set_state_and_learnable_parameters!(mech, z1, μ0)

z1 = storage.z[end-1]
z2 = storage.z[end]
dw2 = zeros(num_state, num_learnable_parameters)

prediction_loss(z2, z2, z1, μ0, idx_parameters, mech)
prediction_jacobian_state!(dz2, z1, μ0, idx_parameters, mech)
prediction_jacobian_parameters!(dw2, z1, μ0, idx_parameters, mech)

z0 = deepcopy(storage.z[1])
ẑ = [deepcopy(storage.z[i+1]) for i=1:H0]
zs = [deepcopy(storage.z[i+1]) for i=1:H0]
ws = [deepcopy(μ0) for i=1:H0]
xtruth = vcat([[deepcopy(storage.z[i+1]); deepcopy(μ0)] for i=1:H0]...)

trajectory_loss(ẑ, zs, z0, ws, idx_parameters, mech)
trajectory_gradient(ẑ, zs, z0, ws, idx_parameters, mech)
trajectory_hessian(ẑ, zs, z0, ws, idx_parameters, mech)


# NonconvexPercival
function local_loss(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_loss(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=1e-3)
end
function local_grad(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_gradient(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=1e-3)
end
function local_hess(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_hessian(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=1e-3)
end
obj_fct = CustomHessianFunction(local_loss, local_grad, local_hess)


function eq_fct(x)
	eq = zeros((H0-1) * nw)
	idx1 = vcat([(i-1)*(nz+nw) + nz .+ (1:nw) for i=1:H0-1]...)
	idx2 = vcat([i*(nz+nw) + nz .+ (1:nw) for i=1:H0-1]...)
	eq = x[idx1] .- x[idx2]
	return 1e-3 .* eq
end

N = H0 * (nz + nw)
lb = -1e3
ub = +1e3
xinit = zeros(N)

model = Nonconvex.Model()
addvar!(model, lb*ones(N), ub*ones(N), init=xinit, integer=falses(N))
add_eq_constraint!(model, eq_fct)

set_objective!(model, obj_fct)

# alg = AugLag()
# options = AugLagOptions()

verbose = true
alg = IpoptAlg()
max_cpu_time = 15.0
print_level = verbose ? 5 : 0
options = IpoptOptions(print_level=print_level, max_cpu_time=max_cpu_time)


xtruth = vcat([[deepcopy(storage.z[i+1]); deepcopy(μ0)] for i=1:H0]...)
result = Nonconvex.optimize(model, alg, xinit, options=options)
xmin = result.minimizer

local_loss(xinit)
local_loss(xtruth)
local_loss(xmin)

plot(xinit)
plot!(xtruth)
plot!(xmin)
plot!(xmin - xtruth)
plot(xmin - xtruth)

xtruth
xmin
xmin[7:7:end]

RobotVisualizer.convert_frames_to_video_and_gif("friction_system_id")
