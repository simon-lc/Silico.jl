include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("prediction.jl")
include("shape_loss.jl")

################################################################################
# visualize
################################################################################
vis = Visualizer()
render(vis)
# open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6);


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
μ0 = [0.1]
ctol = 1e-3
ctol_grad = 1e-3

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
		residual_tolerance=ctol/10,
        complementarity_tolerance=ctol,
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
xp2 =  [+0.00, +1.50, -0.00]
vp15 = [-2.00, +0.00, -3.00]
z0 = [xp2; vp15]
z0 = [xp2; vp15]

H0 = 35
storage = simulate!(mech, z0, H0+1)
vis, anim = visualize!(vis, mech, storage)


################################################################################
# dimensions and parameter selection
################################################################################
# contact parameters =
# 	friction_coefficient = 1     13 .+ (1:1)
# 	Ap = 8     13 .+ (2:9)
# 	bp = 4     13 .+ (10:13)
# 	Ac = 2     13 .+ (14:15)
# 	bc = 1     13 .+ (16:16)

# idx_parameters = 13 .+ [1, 10]
# idx_parameters = 13 .+ [10]
idx_parameters = 13 .+ [10, 11, 12, 13]
num_state = mech.dimensions.body_state
num_parameters = mech.solver.dimensions.parameters
num_learnable_parameters = length(idx_parameters)
nz = num_state
nw = num_learnable_parameters
N = H0 * (nz + nw)

################################################################################
# ground truth
################################################################################
z0 = deepcopy(storage.z[1])
ẑ = [deepcopy(storage.z[i+1]) for i=1:H0]
zs = [deepcopy(storage.z[i+1]) for i=1:H0]
ws = [deepcopy(μ0) for i=1:H0]
# xtruth = vcat([[deepcopy(storage.z[i+1]); deepcopy(μ0)] for i=1:H0]...)
xtruth = vcat([[deepcopy(storage.z[i+1]); deepcopy(μ0); 0.5] for i=1:H0]...)

trajectory_loss(ẑ, zs, z0, ws, idx_parameters, mech)
trajectory_gradient(ẑ, zs, z0, ws, idx_parameters, mech)
trajectory_hessian(ẑ, zs, z0, ws, idx_parameters, mech)


################################################################################
# optimization
################################################################################
# NonconvexPercival
function local_loss(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_loss(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=ctol)
end
function local_grad(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_gradient(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=ctol_grad)
end
function local_hess(x)
	z = [x[(i-1)*(nz+nw) .+ (1:nz)] for i=1:H0]
	w = [x[(i-1)*(nz+nw) + nz .+ (1:nw)] for i=1:H0]
	trajectory_hessian(ẑ, z, z0, w, idx_parameters, mech; complementarity_tolerance=ctol_grad)
end

function eq_fct(x)
	eq = zeros((H0-1) * nw)
	idx1 = vcat([(i-1)*(nz+nw) + nz .+ (1:nw) for i=1:H0-1]...)
	idx2 = vcat([i*(nz+nw) + nz .+ (1:nw) for i=1:H0-1]...)
	eq = x[idx1] .- x[idx2]
	return 1e-3*eq
end

model = Nonconvex.Model()
xinit = vcat([[0*ones(nz); 1.0*ones(nw)] for i = 1:H0]...)
lb = vcat([[-10*ones(nz); 0.0*ones(nw)] for i = 1:H0]...)
ub = vcat([[+10*ones(nz); 1.0*ones(nw)] for i = 1:H0]...)
addvar!(model, lb, ub, init=xinit, integer=falses(N))
add_eq_constraint!(model, eq_fct)
obj_fct = CustomHessianFunction(local_loss, local_grad, local_hess)
set_objective!(model, obj_fct)

verbose = true
alg = IpoptAlg()
max_cpu_time = 90.0
print_level = verbose ? 5 : 0
options = IpoptOptions(print_level=print_level, max_cpu_time=max_cpu_time)

# alg = AugLag()
# options = AugLagOptions(
# 	# first_order = false,
# 	rtol = 1e-4
# 	)

result = Nonconvex.optimize(model, alg, xinit, options=options)
xmin = result.minimizer

local_loss(xinit)
local_loss(xtruth)
local_loss(xmin)
xmin[7:7:end]
xmin[7:8:end]
xmin[8:8:end]
xmin[7:10:end]
xmin[8:10:end]
xmin[9:10:end]
xmin[10:10:end]

# xtruth = vcat([[deepcopy(storage.z[i+1]); deepcopy(μ0); +0.5] for i=1:H0]...)
xtruth = vcat([[deepcopy(storage.z[i+1]); +0.5*ones(nw)] for i=1:H0]...)
# xtruth = vcat([[deepcopy(storage.z[i+1]); +0.5] for i=1:H0]...)
local_loss(xtruth)



a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10
a = 10































################################################################################
# shape
################################################################################
θP = []
for i = 1:H0
	At, bt, ot = transform(A0, b0, o0, storage.x[i][1])
	push!(θP, pack_halfspaces([At, Af], [bt, bf], [ot, of])[1])
end
θP
for i = 1:H0
	At, bt, ot = transform(A0, b0, o0, storage.x[i][1])
	build_2d_polytope!(vis[:polytope], At, bt + At * ot, name=Symbol("poly$i"), color=RGBA(1,1,1,0.2))
end

################################################################################
# camera parameters
################################################################################
nβ = 20
ρ0 = 1e-4
e0 = fill([+1.5, 3.0], H0)
β0 = [-π + atan(e0[i][2] - x[1][2], e0[i][1] - x[1][1]) .+
	Vector(range(+0.08π, -0.08π, length=nβ)) for (i,x) in enumerate(storage.x[1:end-1])]
d0 = [trans_point_cloud(e0[i], β0[i], ρ0, θP[i], polytope_dimensions_p) for i = 1:H0]
for i = 1:H0
	build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(i))
end
set_2d_point_cloud!(vis, e0, d0; name=:point_cloud)

settransform!(vis[:polytope], MeshCat.Translation(-0.05,0,0))

################################################################################
# Initialization
################################################################################
nh = 5
polytope_dimensions = [nh,nh,nh]
np = length(polytope_dimensions)

d_object = filter_point_cloud(d0, poses=[x[1] for x in storage.x], altitude_threshold=0.1)
θinit, kmres = parameter_initialization(d_object, polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)

visualize_kmeans!(vis, θinit, polytope_dimensions, d_object, kmres)
polytope_dimensions
setvisible!(vis[:cluster], true)
setvisible!(vis[:initial], true)
setvisible!(vis[:cluster], false)
setvisible!(vis[:initial], false)


################################################################################
# optimization
################################################################################
# projection
local_projection(θ) = projection(θ, polytope_dimensions,
	Alims=[-1.00, +1.00],
	blims=[+0.02, +0.60],
	olims=[-3.00, +3.00],
	)

shape_loss_parameters
max_iterations = 200

################################################################################
# solve
################################################################################
function local_loss(θ, polytope_dimensions=polytope_dimensions)
	np = length(polytope_dimensions)
	l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	for i = 1:H0
		At, bt, ot = transform(A, b, o, storage.x[i][1])
		for j = 1:np
			build_2d_polytope!(vis[:polytope], At[j], bt[j] + At[j] * ot[j], name=Symbol("poly$i$j"), color=RGBA(0,1,1,0.2))
		end
		θ_t, _ = pack_halfspaces(At, bt, ot)
		l += shape_loss(θ_t, polytope_dimensions, [e0[i]], [β0[i]], ρ0, [d0[i]]; shape_loss_parameters...)
	end
	return l / H0
end

# local_grad(θ) = ForwardDiff.gradient(θ -> local_loss(θ), θ)
θtruth, polytope_dimensions_truth = pack_halfspaces([A0], [b0], [o0])
unpack_halfspaces(θtruth, polytope_dimensions_truth)
local_loss(θtruth, polytope_dimensions_truth)
local_loss(θtruth, polytope_dimensions_truth)

local_loss(θinit)
local_grad(θinit)


adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.a = 3e-3

θsol0, θiter0 = adam_solve!(adam_opt,
	projection=local_projection,
	max_iterations=max_iterations)
vis, anim = visualize_iterates!(vis, θiter0[1:5:end], polytope_dimensions, e0, β0, ρ0,
	max_iterations=max_iterations+1, color=iterate_color)

Asol, bsol, osol = unpack_halfspaces(θsol0, polytope_dimensions)
for i = 1:H0
	for j = 1:np
		build_2d_polytope!(vis[:sol][Symbol(i)], Asol[j], bsol[j] + Asol[j] * osol[j], name=Symbol(j), color=RGBA(1,1,1,1.0))
		set_2d_polytope!(vis[:sol][Symbol(i)], storage.x[i][1][1:2], storage.x[i][1][3:3], name=Symbol(j))
	end
end

for i = 1:H0
	atframe(anim, i) do
		for ii = 1:H0
			setvisible!(vis[:sol][Symbol(ii)], ii == i)
			setvisible!(vis[:point_cloud][Symbol(ii)], ii == i)
		end
	end
end

MeshCat.setanimation!(vis, anim)

# RobotVisualizer.convert_frames_to_video_and_gif("shape_learning_reference")
