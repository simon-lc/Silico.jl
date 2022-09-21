include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
# include("prediction.jl")
include("shape_prediction.jl")
include("shape_loss.jl")

################################################################################
# visualize
################################################################################
vis = Visualizer()
# render(vis)
open(vis)
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
θ_p, polytope_dimensions_p = pack_halfspaces([A0, Af], [b0, bf], [o0, of])
θtruth, polytope_dimensions_truth = pack_halfspaces([A0], [b0], [o0])

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
        sparse_solver=true,
        )
    );
Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +1.50, +0.00]
vp15 = [-2.00, +0.00, -3.00]
z0 = [xp2; vp15]

H0 = 20
storage = simulate!(mech, z0, H0+1)
vis, anim = visualize!(vis, mech, storage)










################################################################################
# visualize true shape
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
max_iterations = 50

################################################################################
# solve
################################################################################
function local_shape_loss(θ, polytope_dimensions=polytope_dimensions)
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

# local_shape_grad(θ) = ForwardDiff.gradient(θ -> local_shape_loss(θ), θ)
local_shape_loss(θtruth, polytope_dimensions_truth)
local_shape_loss(θinit)
local_shape_grad(θinit)


adam_opt = Adam(θinit, local_shape_loss, local_shape_grad)
adam_opt.a = 5e-3

θsol0, θiter0 = adam_solve!(adam_opt,
	projection=local_projection,
	max_iterations=max_iterations)
vis, anim = visualize_iterates!(vis, θiter0[1:2:end], polytope_dimensions, e0, β0, ρ0,
	max_iterations=max_iterations+1, color=iterate_color)

Asol, bsol, osol = unpack_halfspaces(θsol0, polytope_dimensions)



RobotVisualizer.convert_frames_to_video_and_gif("learning_shape_second_phase_50")

################################################################################
# dynamics_loss
################################################################################
learned_mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=μ0[1],
	method_type=:symbolic,
	A=Asol,
	b=bsol .+ Asol .* osol,
    options=Mehrotra.Options(
        verbose=false,
		residual_tolerance=ctol/10,
        complementarity_tolerance=ctol,
		compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        )
    );

parameter_dimension.(learned_mech.contacts)
idx_parameters = vcat([13 + (3nh+4)*(i-1) .+ (1+1:1+3nh) for i=1:np]...)
num_state = mech.dimensions.body_state
num_learnable_parameters = length(idx_parameters)
nz = num_state
nw = num_learnable_parameters


################################################################################
# test simulation
################################################################################
function dynamics_loss(learned_mechanism::Mechanism{T}, storage::TraceStorage{T,N}, w, idx_parameters) where {T,N}
	H = N - 1
	nz = learned_mechanism.dimensions.state
	z_pred = zeros(nz)
	Q = I

	l = 0.0
	for i = 1:H
		z = storage.z[i]
		z1 = storage.z[i+1]
		u = zeros(learned_mechanism.dimensions.input)
		dynamics(z_pred, learned_mechanism, z, u, w=w, idx_parameters=idx_parameters)

		l += 0.5 * (z_pred - z1)' * Q * (z_pred - z1)
	end
	return l / H
end

function dynamics_grad(learned_mechanism::Mechanism{T}, storage::TraceStorage{T,N}, w, idx_parameters) where {T,N}
	H = N - 1
	nz = learned_mechanism.dimensions.state
	nw = length(idx_parameters)
	dw = zeros(nz, nw)
	grad = zeros(nw)
	z_pred = zeros(nz)
	Q = I

	for i = 1:H
		z = storage.z[i]
		z1 = storage.z[i]
		u = zeros(learned_mechanism.dimensions.input)
		dynamics(z_pred, learned_mechanism, z, u, w=w, idx_parameters=idx_parameters)
		dynamics_jacobian_parameters(dw, learned_mechanism, z, u, w=w, idx_parameters=idx_parameters)
		grad .+= dw' * Q * (z_pred - z1)
	end
	return grad / H
end

function θ_to_w(θ::Vector{T}, polytope_dimensions) where T
	np = length(polytope_dimensions)
	w = zeros(T, sum(3*polytope_dimensions))

	off = 0
	for i = 1:np
		nh = polytope_dimensions[i]
		A, b, o = unpack_halfspaces(θ, polytope_dimensions, i)
		w[off .+ (1:3nh)] = [A; b + reshape(A, (nh,2)) * o]
		off += 3nh
	end
	return w
end

function θ_to_w_jacobian(θ, polytope_dimensions)
	ForwardDiff.jacobian(θ -> θ_to_w(θ, polytope_dimensions), θ)
end




# storage
# w = θ_to_w(θtruth, polytope_dimensions_truth)
# idx_parameters = vcat([13 + (12+4)*(i-1) .+ (1+1:1+12) for i=1:1]...)
# dynamics_loss(mech, storage, w, idx_parameters)

w = θ_to_w(θsol0, polytope_dimensions)
idx_parameters = vcat([13 + (3nh+4)*(i-1) .+ (1+1:1+3nh) for i=1:np]...)
dynamics_loss(learned_mech, storage, w, idx_parameters)
g10 = dynamics_grad(learned_mech, storage, w, idx_parameters)
# g20 = FiniteDiff.finite_difference_gradient(w -> dynamics_loss(learned_mech, storage, w, idx_parameters), w)
scatter(g10)
scatter!(g20)

function local_dynamics_loss(θ, polytope_dimensions=polytope_dimensions)
	w = θ_to_w(θ, polytope_dimensions)
	l = dynamics_loss(learned_mech, storage, w, idx_parameters)
	return l
end

function local_dynamics_grad(θ, polytope_dimensions=polytope_dimensions)
	w = θ_to_w(θ, polytope_dimensions)
	dwdθ = θ_to_w_jacobian(θ, polytope_dimensions)
	dldw = dynamics_grad(learned_mech, storage, w, idx_parameters)
	grad = dwdθ' * dldw
	return grad
end



function local_loss(θ, polytope_dimensions=polytope_dimensions)
	l = 0.0
	l += local_shape_loss(θ, polytope_dimensions)
	l += local_dynamics_loss(θ, polytope_dimensions)
	return l
end

function local_grad(θ, polytope_dimensions=polytope_dimensions)
	nθ = length(θ)
	grad = zeros(nθ)
	grad .+= local_shape_grad(θ)
	grad .+= local_dynamics_grad(θ, polytope_dimensions)
	return grad
end


local_loss(θsol0)
local_grad(θsol0)


adam_opt = Adam(deepcopy(θsol0), local_loss, local_grad)
adam_opt.a = 2e-3

θsol1, θiter1 = adam_solve!(adam_opt,
	projection=local_projection,
	max_iterations=max_iterations)
vis, anim = visualize_iterates!(vis, θiter1[1:2:end], polytope_dimensions, e0, β0, ρ0,
	max_iterations=max_iterations+1, color=iterate_color)
