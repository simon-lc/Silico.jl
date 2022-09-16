include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include("../flux/shape_loss.jl")


vis = Visualizer()
# open(vis)
render(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, color=RGBA(0.4,0.4,0.4,0.4))
iterate_color = RGBA(1,1,0,0.6);

Ap0 = [
	+1.0 +0.3;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.3 -1.0;
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
Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
vp15 = [-0,0,-0.3*9.0]
xp2 = [+0.0,2.00,+0.00]
z0 = [xp2; vp15]

H0 = 25
storage = simulate!(mech, z0, H0)
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
settransform!(vis[:polytope], MeshCat.Translation(-0.05,0,0))
setvisible!(vis[:polytope], true)
setvisible!(vis[:polytope], false)

################################################################################
# camera parameters
################################################################################
nβ = 20
poses = [x[1] for x in storage.x]
x = [p[1:2] for p in poses] # position
θ = [p[3] for p in poses] # orientation
bRw = [[cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])] for i=1:H0]
eye_positions = [[0.0, 3.0] for i=1:H0]
# angles = [-π + atan(eye_positions[i][2] - x[1][2], eye_positions[i][1] - x[1][1]) .+
# 	Vector(range(+0.12π, -0.12π, length=nβ)) for (i,x) in enumerate(storage.x)]
angles = [-π/2 .+ Vector(range(+0.2π, -0.2π, length=nβ)) for (i,x) in enumerate(storage.x)]

α, α_hit, αmax, αmax_hit, v, v_hit, e, e_hit = vectorized_ray(eye_positions, angles, [Ap0], [bp0], [op0], poses;
	altitude_threshold=0.01,
	max_length=50.00,
	)

for i = 1:H0
	build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(i))
end

d0b = [e[:,(i-1)*nβ .+ (1:nβ)] + α[(i-1)*nβ .+ (1:nβ)]' .* v[:,(i-1)*nβ .+ (1:nβ)] for i = 1:H0]
d0w = [x[i] .+ bRw[i]' * d0b[i] for i = 1:H0]


d0_hit = e_hit .+ α_hit' .* v_hit
d0 = e .+ α' .* v
set_2d_point_cloud!(vis, eye_positions, d0b; name=:point_cloud)
set_2d_point_cloud!(vis, eye_positions, d0w; name=:point_cloud)


################################################################################
# Initialization
################################################################################
nh = 5
polytope_dimensions = [nh,nh,nh]
np = length(polytope_dimensions)

θinit, kmres = parameter_initialization(d0_hit, polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d0_hit, kmres)
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
	blims=[+0.05, +0.60],
	olims=[-3.00, +3.00],
	)

parameters = Dict(
	:thickness => 0.2,
	:δ_sdf => 0.025,
	:δ_sigmoid => 0.01,
	:δ_softabs => 0.5,
	:altitude_threshold => 0.01,
	:rendering => 1.0 * 5.0,
	:sdf_matching => 1.0 * 10.0,
	:side_regularization => 1.0 * 0.5,
	:shape_regularization => 0000.0 * 0.5,
	:inside => 1.0 * 0.4,
	:outside => 1.0 * 0.1,
	:floor => 1.0 * 0.1,
	)
max_iterations = 20

################################################################################
# solve
################################################################################

# loss and gradients
function local_loss(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	l = shape_loss(
		α, α_hit, αmax, αmax_hit, v, v_hit, e, e_hit,
		A, b, bo; parameters...
		)
	return l
end

shape_loss_gradient(
	α, α_hit, αmax, αmax_hit, v, v_hit, e, e_hit,
	A, b, bo, parameters,
	) = gradient(keyword_shape_loss,
		α, α_hit, αmax, αmax_hit, v, v_hit, e, e_hit,
		A, b, bo, parameters,
		)[9:11]

function local_grad(θ)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	grads = shape_loss_gradient(
		α, α_hit, αmax, αmax_hit, v, v_hit, e, e_hit,
		A, b, bo, parameters,
		)
	dldAb = [vcat(vec.(grads[1])...); vcat(grads[2]...); vcat(grads[3]...)]
	dAbdθ = ForwardDiff.jacobian(θ -> preprocess_halfspaces(θ, polytope_dimensions, vectorize=true), θ)
	return  dAbdθ' * dldAb
end


local_loss(θinit)
local_loss(θsol0)
local_grad(θinit)

################################################################################
# solve
################################################################################
adam_opt = Adam(θinit, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 1e-2
max_iterations = 40
@elapsed θsol0, θiter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=10max_iterations)

visualize_iterates!(vis, θiter0[1:10:end], polytope_dimensions, eye_positions[1],
 	angles, 1e-4, max_iterations=max_iterations+1, color=iterate_color)
Asol, bsol, osol = unpack_halfspaces(local_projection(θsol0), polytope_dimensions)
