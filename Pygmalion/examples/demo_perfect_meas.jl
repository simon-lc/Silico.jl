using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using ForwardDiff
using Clustering
using LinearAlgebra

include("../src/DojoLight.jl")
include("../system_identification/sr1.jl")
include("../system_identification/adam.jl")
include("halfspace.jl")
include("transparency_point_cloud.jl")
include("visuals.jl")
include("softmax.jl")
include("utils.jl")

vis = Visualizer()
# render(vis)
open(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6)

Ap0 = [
	+1.0 +0.3;
	+0.0 +1.0;
	-1.0 +0.3;
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

# build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0, name=:poly0, color=RGBA(1,1,1,1.0))

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
    method_type=:symbolic,
	A=Ap0,
	b=bp0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );
solve!(mech.solver)

################################################################################
# test simulation
################################################################################
# xp2 = [+0.0,1.50,-0.25]
vp15 = [-0,0,-9.0]
# z0 = [xp2; vp15]
xp2 = [+0.0,1.50,-0.00]
# vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]

H0 = 15
u0 = [0.0, 0.0, 0.0]
ctrl = open_loop_controller([u0])

storage = simulate!(mech, z0, H0, controller=ctrl)
vis, anim = visualize!(vis, mech, storage)
storage.x

θP = []
for i = 1:H0
	At, bt, ot = transform(Ap0, bp0, op0, storage.x[i][1])
	push!(θP, pack_halfspaces([At, Af], [bt, bf], [ot, of])[1])
end
θP
for i = 1:H0
	At, bt, ot = transform(Ap0, bp0, op0, storage.x[i][1])
	build_2d_polytope!(vis[:polytope], At, bt + At * ot, name=Symbol("poly$i"), color=RGBA(1,1,1,0.2))
end

################################################################################
# camera parameters
################################################################################
nβ = 10
ρ0 = 1e-4
e0 = fill([0.0, 3.0], H0)
β0 = [-π + atan(e0[i][2] - x[1][2], e0[i][1] - x[1][1]) .+ Vector(range(+0.20π, -0.20π, length=nβ)) for (i,x) in enumerate(storage.x)]
d0 = [trans_point_cloud(e0[i], β0[i], ρ0, θP[i], polytope_dimensions_p) for i = 1:H0]
for i = 1:H0
	build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(i))
end
set_2d_point_cloud!(vis, e0, d0; name=:point_cloud)

settransform!(vis[:polytope], MeshCat.Translation(-0.05,0,0))

################################################################################
# Initialization
################################################################################
nh = 6
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

parameters = Dict(
	:thickness => 0.2,
	:δ_sdf => 15.0,
	:δ_sigmoid => 0.1,
	:δ_softabs => 0.5,
	:altitude_threshold => 0.01,
	:rendering => 5.0,
	:sdf_matching => 20.0,
	:overlap => 2.0,
	:individual => 1.0,
	:side_regularization => 0.5,
	:shape_regularization => 0.5,
	:inside => 1.0,
	:outside => 0.1,
	:floor => 0.1,
)
max_iterations = 20

################################################################################
# solve
################################################################################

# parameters = Dict(
# 	:thickness => 0.2,
# 	:δ_sdf => 15.0,
# 	:δ_sigmoid => 0.1,
# 	:δ_softabs => 0.5,
# 	:altitude_threshold => 0.01,
# 	:rendering => 1.0 * 5.0,
# 	:sdf_matching => 1.0 * 20.0,
# 	:overlap => 1.0 * 2.0,
# 	:individual => 1.0 * 1.0,
# 	:side_regularization => 1.0 * 0.5,
# 	:shape_regularization => 1.0 * 0.5,
# 	:inside => 1.0 * 1.0,
# 	:outside => 1.0 * 0.1,
# 	:floor => 1.0 * 0.1,
# )
function local_loss(θ)
	l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	for i = 1:H0
		At, bt, ot = transform(A, b, o, storage.x[i][1])
		build_2d_polytope!(vis[:polytope], At[1], bt[1] + At[1] * ot[1], name=Symbol("poly$i"), color=RGBA(0,1,1,0.2))
		θ_t, _ = pack_halfspaces(At, bt, ot)
		l += shape_loss(θ_t, polytope_dimensions, [e0[i]], [β0[i]], ρ0, [d0[i]]; parameters...)
	end
	return l
end

# local_grad(θ) = ForwardDiff.gradient(θ -> local_loss(θ), θ)
θtruth = pack_halfspaces(Ap0, bp0, op0)
local_loss(θtruth)
local_loss(θinit)
local_grad(θinit)


adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 3e-3
θsol0, θiter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=10max_iterations)
vis, anim = visualize_iterates!(vis, θiter0[1:5:end], polytope_dimensions, e0, β0, ρ0, max_iterations=1max_iterations+1, color=iterate_color)

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
