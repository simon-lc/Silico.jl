include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))

vis = Visualizer()
open(vis)
# render(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, x = 0.1, color=RGBA(0.4,0.4,0.4,0.4))
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
vp15 = [-0,0,-0.6*9.0]
xp2 = [+0.0,2.00,+0.00]
z0 = [xp2; vp15]

H0 = 25
storage = simulate!(mech, z0, H0)
vis, anim = visualize!(vis, mech, storage, show_contact=false)

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
eye_positions = [[0.0, 3.0] for i=1:H0]
angles = [-π/2 .+ Vector(range(+0.2π, -0.2π, length=nβ)) for i = 1:H0]

poses = [x[1] for x in storage.x]
x = [p[1:2] for p in poses] # position
θ = [p[3] for p in poses] # orientation
bRw = [[cos(θ[i]) sin(θ[i]); -sin(θ[i]) cos(θ[i])] for i=1:H0]
noise = [[0.2, 0.2, 0.5] .* (rand(3) .- 0.5) for x in storage.x]
noisy_poses = [storage.x[i][1] + noise[i]  for i = 1:H0]


α, αmax, v, e, hit_indices = vectorized_ray(eye_positions, angles, [Ap0], [bp0], [op0], poses;
	altitude_threshold=0.01,
	max_length=50.00,
	)
v_noisy, e_noisy = noise_transform(v, e, noise)

d0b = e .+ α' .* v
d0b_noisy = e_noisy .+ α' .* v_noisy
d0b_split = [d0b[:,(i-1)*nβ .+ (1:nβ)] for i=1:H0]
d0b_noisy_split = [d0b_noisy[:,(i-1)*nβ .+ (1:nβ)] for i=1:H0]
d0w_split = [x[i] .+ bRw[i]' * d0b_split[i] for i = 1:H0]

for i = 1:H0
	build_point_cloud!(vis[:point_cloud], nβ; color=RGBA(0.9,0.1,0.1,1), name=Symbol(i))
end
set_2d_point_cloud!(vis, eye_positions, d0b_noisy_split; name=:point_cloud)
set_2d_point_cloud!(vis, eye_positions, d0b_split; name=:point_cloud)
set_2d_point_cloud!(vis, eye_positions, d0w_split; name=:point_cloud)


################################################################################
# Initialization
################################################################################
nh = 5
polytope_dimensions = [nh,nh,nh]
np = length(polytope_dimensions)

θinit, kmres = parameter_initialization(d0b_noisy[:,hit_indices], polytope_dimensions)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d0b_noisy[:,hit_indices], kmres)
setvisible!(vis[:cluster], true)
setvisible!(vis[:initial], true)
setvisible!(vis[:cluster], false)
setvisible!(vis[:initial], false)

################################################################################
# optimization
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

# projection
function local_projection(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)

	θπ = projection(θ, polytope_dimensions,
		Alims=[-1.00, +1.00],
		blims=[+0.05, +0.60],
		olims=[-3.00, +3.00],
		)
	return pack_poses_halfspace_variables(θ, denoise)
end

denoise_init = [zeros(3) for i = 1:H0]
vars_init = pack_poses_halfspace_variables(θinit, denoise_init)
unpack_poses_halfspace_variables(vars_init, polytope_dimensions)
local_projection(vars_init)

# loss and gradients
function local_loss(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)

	l = noisy_shape_loss(
		α, αmax, v, e, hit_indices,
		A, b, bo,
		noise, denoise,
		ShapeLossOptions1200(),
		)
	return l
end

function local_grad(vars)
	θ, denoise = unpack_poses_halfspace_variables(vars, polytope_dimensions)
	A, b, bo = preprocess_halfspaces(θ, polytope_dimensions)

	grads = noisy_shape_gradient(
		α, αmax, v, e, hit_indices,
		A, b, bo,
		noise, denoise,
		ShapeLossOptions1200(),
		)

	dldAb = [vcat(vec.(grads[1])...); vcat(grads[2]...); vcat(grads[3]...)]
	dlddenoise = vcat(grads[4]...)
	dAbdθ = ForwardDiff.jacobian(θ -> preprocess_halfspaces(θ, polytope_dimensions, vectorize=true), θ)
	return [dAbdθ' * dldAb; dlddenoise]
end


local_loss(vars_init)
local_loss(vars_sol0)
local_grad(vars_init)



################################################################################
# solve
################################################################################
adam_opt = Adam(vars_init, local_loss, local_grad)
adam_opt.eps = 1e-8
adam_opt.a = 5e-3
max_iterations = 400
visual_iterations = 50

subset = 1:Int(ceil(max_iterations/visual_iterations)):max_iterations
@elapsed vars_sol0, vars_iter0 = adam_solve!(adam_opt, projection=local_projection, max_iterations=max_iterations)
θsol0, denoise_sol0 = unpack_poses_halfspace_variables(vars_sol0, polytope_dimensions)

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

# convert_frames_to_video_and_gif("denoising_shape_learning")
