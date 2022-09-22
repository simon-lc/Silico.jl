include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))
include(joinpath(@__DIR__, "..", "implicit_dynamics", "prediction.jl"))


vis = Visualizer()
open(vis)
# render(vis)
set_background!(vis)
# set_light!(vis, direction="Negative")
set_light!(vis, direction="Positive")
set_floor!(vis, x = 0.1, color=RGBA(0.4,0.4,0.4,0.4))
set_floor!(vis)
iterate_color = RGBA(1,1,0,0.6);

################################################################################
# inertial parameters
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.1,
    method_type=:symbolic,
	A=A_recipient,
	b=bo_recipient,
    options=Mehrotra.Options(
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        sparse_solver=false,
        )
    );
Mehrotra.solve!(mech.solver)

################################################################################
# test simulation
################################################################################
xp2 =  [-0.50, 1.00, +1.00]
vp15 = [+3.00, 0.00, -4.00]
z0 = [xp2; vp15]

horizon = 25
storage = simulate!(mech, z0, horizon)
vis, anim = visualize!(vis, mech, storage, show_contact=false)

################################################################################
# camera parameters
################################################################################
nβ = 40
eye_positions = [[0.0, 3.0] for i=1:horizon]
angles = [-π/2 .+ Vector(range(+0.2π, -0.2π, length=nβ)) for i = 1:horizon]

poses = [x[1] for x in storage.x]
x = [p[1:2] for p in poses] # position
θ = [p[3] for p in poses] # orientation
bRw = [[cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])] for i=1:horizon]
noise = [1*[0.1, 0.1, 0.25] .* (rand(3) .- 0.5) for x in storage.x]
noisy_poses = [storage.x[i][1] + noise[i]  for i = 1:horizon]


α, αmax, v, e, hit_indices = vectorized_ray(eye_positions, angles,
	A_recipient, b_recipient, o_recipient, poses;
	altitude_threshold=0.01,
	max_length=50.00,
	)
v_noisy, e_noisy = noise_transform(v, e, noise)

d0b = e .+ α' .* v
d0b_noisy = e_noisy .+ α' .* v_noisy
d0b_split = [d0b[:,(i-1)*nβ .+ (1:nβ)] for i=1:horizon]
d0b_noisy_split = [d0b_noisy[:,(i-1)*nβ .+ (1:nβ)] for i=1:horizon]
d0w_split = [x[i] .+ bRw[i]' * d0b_split[i] for i = 1:horizon]

for i = 1:horizon
	build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(i))
end
set_2d_point_cloud!(vis, eye_positions, d0b_noisy_split; name=:point_cloud)
set_2d_point_cloud!(vis, eye_positions, d0b_split; name=:point_cloud)
set_2d_point_cloud!(vis, eye_positions, d0w_split; name=:point_cloud)


################################################################################
# Initialization
################################################################################
nh = 5
polytope_dimensions = [nh,nh,nh,nh]
np = length(polytope_dimensions)

θinit, kmres = parameter_initialization(d0b_noisy[:,hit_indices], polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d0b_noisy[:,hit_indices], kmres)
setvisible!(vis[:cluster], true)
setvisible!(vis[:initial], true)
setvisible!(vis[:cluster], false)
setvisible!(vis[:initial], false)


################################################################################
# Prediction loss
################################################################################
learned_mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.1,
    method_type=:symbolic,
	A=deepcopy(A_sol0),
	b=deepcopy(bo_sol0),
    options=Mehrotra.Options(
		verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        sparse_solver=false,
        )
    );
Mehrotra.solve!(learned_mech.solver)

################################################################################
# test traj loss
################################################################################
idx_parameters = vcat([c.index.parameters[2:16] for c in learned_mech.contacts]...) # Ap, bp

ẑ_ref = [deepcopy(storage.z[i]) for i = 2:horizon]
ẑ_traj = [[noisy_poses[i]; (noisy_poses[i] - noisy_poses[i-1]) / timestep] for i=2:horizon]

plot(hcat(ẑ_ref...)')
plot(hcat(ẑ_traj...)')

prediction_loss(ẑ_ref[1], ẑ_ref[1], z0, w_init[1], idx_parameters, learned_mech;
	complementarity_tolerance=1e-3, Q=I, R=I)

trajectory_loss(ẑ_traj, z_traj, z0, w_init, idx_parameters, learned_mech;
	complementarity_tolerance=1e-3, Q=I, R=I)

trajectory_gradient(ẑ_traj, z_traj, z0, w_init, idx_parameters, learned_mech;
	complementarity_tolerance=1e-3, Q=I, R=0.01*I)




################################################################################
# packing
################################################################################
function pack_poses_halfspace_variables(θ, denoise)
	return [θ; vcat(denoise...)]
end

function unpack_poses_halfspace_variables(vars, polytope_dimensions)
	np = length(polytope_dimensions)
	nθ = 3 * sum(polytope_dimensions) + 2 * np
	nv = length(vars)
	H = Int((nv - nθ) / 3)

	θ = vars[1:nθ]
	denoise = [vars[nθ + 3*(i-1) .+ (1:3)] for i = 1:H]
	return θ, denoise
end

function preprocess_vars(vars, polytope_dimensions; vectorize=false)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	wi = vcat([[vec(A[i]); bo[i]] for i = 1:np]...)
	w_init = [wi for i=2:horizon]

	denoised_poses = deepcopy(noisy_poses) .- denoise
	z_traj = [[denoised_poses[i]; (denoised_poses[i] - denoised_poses[i-1]) / timestep] for i=2:horizon]

	traj = vcat([[z_traj[i]; w_init[i]] for i=1:horizon-1]...)
	vectorize && return traj
	return w_init, z_traj
end

################################################################################
# optimization
################################################################################
# projection
function local_projection(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)

	θπ = projection(θ, polytope_dimensions,
		Alims=[-1.00, +1.00],
		blims=[+0.05, +0.60],
		olims=[-3.00, +3.00],
		)
	return pack_poses_halfspace_variables(θπ, denoise)
end

# loss and gradients
function local_loss(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	w_init, z_traj = preprocess_vars(vars, polytope_dimensions)

	l = noisy_shape_loss(
		α, αmax, v, e, hit_indices,
		A, b, bo,
		noise, denoise,
		ShapeLossOptions1200(),
		)

	lt = trajectory_loss(ẑ_traj, z_traj, z0, w_init, idx_parameters, learned_mech;
		complementarity_tolerance=1e-3, Q=Diagonal([300ones(3); 1e-1ones(3)]), R=1e-15I)
	l += lt
	@show lt

	return l
end

function local_grad(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)
	w_init, z_traj = preprocess_vars(vars, polytope_dimensions)

	grads = noisy_shape_gradient(
		α, αmax, v, e, hit_indices,
		A, b, bo,
		noise, denoise,
		ShapeLossOptions1200(),
		)

	dldAb = [vcat(vec.(grads[1])...); vcat(grads[2]...); vcat(grads[3]...)]
	dlddenoise = vcat(grads[4]...)
	dAbdθ = ForwardDiff.jacobian(θ -> preprocess_halfspaces(θ, polytope_dimensions, vectorize=true), θ)
	dl = [dAbdθ' * dldAb; dlddenoise]


	dldxw = trajectory_gradient(ẑ_traj, z_traj, z0, w_init, idx_parameters, learned_mech;
		complementarity_tolerance=1e-3, Q=Diagonal([300ones(3); 1e-1ones(3)]), R=1e-15I)
	dxwdvars = ForwardDiff.jacobian(vars -> preprocess_vars(vars, polytope_dimensions, vectorize=true), vars)
	dlt = dxwdvars' * dldxw
	dl += dlt
	return dl
end



################################################################################
# test
################################################################################
denoise_init = [zeros(3) for i = 1:horizon]
vars_init = pack_poses_halfspace_variables(θinit, denoise_init)

local_loss(vars_init)
local_loss(vars_sol0)
local_grad(vars_init)

################################################################################
# solve
################################################################################
adam_opt = Adam(vars_init, local_loss, local_grad)
adam_opt.a = 5e-3
max_iterations = 400
visual_iterations = 25

subset = 1:Int(ceil(max_iterations/visual_iterations)):max_iterations
@elapsed vars_sol0, vars_iter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=max_iterations)
θsol0, denoise_sol0 = unpack_poses_halfspace_variables(vars_sol0, polytope_dimensions)
A_sol0, b_sol0, o_sol0 = unpack_halfspaces(θsol0, polytope_dimensions)
A_sol0, b_sol0, bo_sol0 = preprocess_halfspaces(θsol0, polytope_dimensions)
denoised_poses_sol0 = noisy_poses .- denoise_sol0



vis, anim = visualize_iterates!(vis, vars_iter0[subset], polytope_dimensions, eye_positions[1],
 	angles, 1e-4, max_iterations=visual_iterations, color=iterate_color)

plot(hcat(noise...)'[:,1], color=:red)
plot!(hcat(denoise_sol0...)'[:,1], color=:blue)

plot(hcat(noise...)'[:,2], color=:red)
plot!(hcat(denoise_sol0...)'[:,2], color=:blue)

plot(hcat(noise...)'[:,3], color=:red)
plot!(hcat(denoise_sol0...)'[:,3], color=:blue)


################################################################################
# visualize denoising
################################################################################
num_hit = sum(hit_indices)
build_point_cloud!(vis[:hit_cloud], num_hit; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
settransform!(vis[:hit_cloud], MeshCat.Translation(0.3, 0.0, 0.0))
for (j,i) in enumerate(subset)
	_, denoise_i = unpack_poses_halfspace_variables(vars_iter0[i], polytope_dimensions)
	vi, ei = noise_transform(v, e, noise - denoise_i)
	di = ei .+ α' .* vi
	di_hit = di[:,hit_indices]
	atframe(anim, j) do
		set_2d_point_cloud!(vis, eye_positions[1:1], [di_hit], name=:hit_cloud)
	end
end
MeshCat.setanimation!(vis, anim)







################################################################################
# test learned convex bundle
################################################################################
for i = 1:np
	shape = learned_mech.bodies[1].shapes[i]
	shape.A .= A_sol0[i]
	shape.b .= b_sol0[i]
	shape.o .= o_sol0[i]
	contact = learned_mech.contacts[i]
	contact.A_parent_collider .= A_sol0[i]
	contact.b_parent_collider .= bo_sol0[i]
end
mech.bodies[1]
horizon = 25
w_sol0, _ = preprocess_vars(vars_iter0[end], polytope_dimensions; vectorize=false)
learned_mech.solver.parameters[idx_parameters] .= w_sol0[1]
update_nodes!(learned_mech)
learned_storage = simulate!(learned_mech, z0, horizon)
vis, anim = visualize!(vis, learned_mech, learned_storage, show_contact=false, color=RGBA(0,0.4,0.9,0.4), name=:learned)
vis, anim = visualize!(vis, mech, storage, show_contact=false, color=RGBA(1,1,1,1), name=:robot, animation=anim)

convert_frames_to_video_and_gif("forward_sim_normal")
