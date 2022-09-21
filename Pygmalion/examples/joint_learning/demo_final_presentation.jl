include(joinpath(module_dir(), "Pygmalion/Pygmalion.jl"))

vis = Visualizer()
open(vis)
# render(vis)
set_background!(vis)
set_light!(vis, direction="Negative")
set_floor!(vis, x = 0.1, color=RGBA(0.4,0.4,0.4,0.4))
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

horizon = 25
storage = simulate!(mech, z0, horizon)
vis, anim = visualize!(vis, mech, storage, show_contact=false)

# for i = 1:horizon
# 	At, bt, ot = transform(Ap0, bp0, op0, storage.x[i][1])
# 	build_2d_polytope!(vis[:polytope], At, bt + At * ot, name=Symbol("poly$i"), color=RGBA(1,1,1,0.2))
# end
# settransform!(vis[:polytope], MeshCat.Translation(-0.05,0,0))
# setvisible!(vis[:polytope], true)
# setvisible!(vis[:polytope], false)

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
polytope_dimensions = [nh,nh,nh,nh,nh]
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

denoise_init = [zeros(3) for i = 1:horizon]
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
adam_opt.a = 5e-3
max_iterations = 400
visual_iterations = 25

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
# convert_frames_to_video_and_gif("u_shape_denoising")






















# ## visualizer
vis = Visualizer()
# render(vis)
open(vis)
RobotVisualizer.set_background!(vis)
RobotVisualizer.set_light!(vis)
RobotVisualizer.set_floor!(vis, x=0.1)
RobotVisualizer.set_camera!(vis, zoom=3.0)

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
    # method_type=:finite_difference,
	A=A_box[1],
	b=bo_box[1],
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
vp15 = [+3.0, +0.0, +2.0]
xp2  = [+0.0, +0.75, +0.0]
z0 = [xp2; vp15]

storage = simulate!(mech, z0, horizon)
vis, anim = visualize!(vis, mech, storage)

configuration_ref = [[storage.x[i][1]; storage.v[i][1]] for i = 1:horizon]
JLD2.save(joinpath(@__DIR__, "../deps/configuration_ref_$horizon.jld2"), "configuration_ref", configuration_ref)

################################################################################
# initial guess
################################################################################
Ap1 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
bp1 = 0.75*[
	+1,
	+1,
	+1,
	+1,
	]
friction_coefficient1 = 0.5

configuration_init = [[deepcopy(z0); zeros(12+13)] for i = 1:horizon]
for i = 2:horizon
	mech.solver.parameters[14] = friction_coefficient1
	mech.solver.parameters[end-12-3+1:end-3] .= [vec(Ap1); bp1]
	update_nodes!(mech)

	storage_init = simulate!(mech, configuration_init[i-1][1:6], 2)
	configuration_init[i] .= [
		deepcopy(storage_init.x[2][1]);
		deepcopy(storage_init.variables[1][1:15]);
		deepcopy(vec(Ap1));
		deepcopy(bp1);
		friction_coefficient1;
		]
end

configuration_init[1][7:end] .= configuration_init[2][7:end]
JLD2.save(joinpath(@__DIR__, "../deps/configuration_init_$horizon.jld2"), "configuration_init", configuration_init)

plot(hcat(configuration_init...)'[:,1:3])


# convert_frames_to_video_and_gif("sysid_drop_and_slide_sol")
plot(hcat(x_sol...)'[:,1:3])
plot(hcat(configuration_init...)'[:,1:3])

plot(hcat(x_sol...)'[:,4:6])
plot(hcat(configuration_init...)'[:,4:6])

plot(hcat(x_sol...)'[:,7:end])
plot(hcat(configuration_init...)'[:,7:end])




















# ## dimensions
nh = 4 # number of halfspaces
nx = sum([3 3 2 1 1 1 2 nh 1 2nh nh 1]) # p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap, bp, friction_coefficient
nu = 0

num_states = [nx for t = 1:horizon]
num_actions = [nu for t = 1:horizon-1]

function unpack_state(x; nh=4)
    off = 0
    p3 = x[off .+ (1:3)]; off += 3
    v25 = x[off .+ (1:3)]; off += 3
    c = x[off .+ (1:2)]; off += 2
    ϕ = x[off .+ (1:1)]; off += 1

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    λp = x[off .+ (1:nh)]; off += nh
    λc = x[off .+ (1:1)]; off += 1

    Ap = x[off .+ (1:2nh)]; off += 2nh
    Ap = reshape(Ap, (nh,2))
    bp = x[off .+ (1:nh)]; off += nh
    friction_coefficient = x[off .+ (1:1)]; off +=1

    # 3 3 2 1 1 2 2 nh 1 2nh nh 1
    return p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap, bp, friction_coefficient
end

function polytope_dynamics(y, x, u;
        timestep=0.05,
        gravity=-10.0,
        mass=1.0,
        inertia=0.2,
        # friction_coefficient = 0.1,
        Ac=[0 1.0],
        bc=[0.0],
        )

    # unpack
    p2, v15, _, _, _, _, _, _, _, Ap2, bp2, friction_coefficient2 = unpack_state(x)
    p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap3, bp3, friction_coefficient3 = unpack_state(y)

    # integrator
    p1 = p2 - timestep * v15
    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # contact points
    pp3 = p3
    pc3 = zeros(3)

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap3' * λp
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]

    res = [
        # integrator
        p2 + timestep * v25 - p3;
        # dynamics
        M ./ timestep * (p3 - 2p2 + p1) - timestep .* [0; mass * gravity; 0] - wrench_p;
        # contact optimality
        x_2d_rotation(pp3[3:3]) * Ap3' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc);
        # parameter consistency
        vec(Ap2 - Ap3);
        bp2 - bp3;
        friction_coefficient2 - friction_coefficient3;
    ]
    return res
end

function slackness(x;
        Ac=[0 1.0],
        bc=[0.0],
        )

    # unpack
    p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap3, bp3, friction_coefficient3 = unpack_state(x)

    # contact points
    pp3 = p3
    pc3 = zeros(3)

    # contact position in the world frame
    contact_w = c + pp3[1:2]
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap3' * λp
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # tangential velocities at the contact point
    tanvel_p = v25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); v25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    return [
        ϕ;
        (friction_coefficient3[1] * γ - [sum(β)]);
        ([+tanvel; -tanvel] + ψ[1]*ones(2));
        (- Ap3 * contact_p + bp3 + ϕ .* ones(nh));
        (- Ac * contact_c + bc + ϕ .* ones(1));
    ]
end

function contact_constraints_equality_t(x, u,;
        Ac=[0 1.0],
        bc=[0.0],
        )

	# unpack
    p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap3, bp3, friction_coefficient3 = unpack_state(x)
    return [γ; ψ; β; λp; λc] .* slackness(x, Ac=Ac, bc=bc)
end

function contact_constraints_inequality_t(x, u;
        Ac=[0 1.0],
        bc=[0.0],
        )

	# unpack
	p3, v25, c, ϕ, γ, ψ, β, λp, λc, Ap3, bp3, friction_coefficient3 = unpack_state(x)
    return [
        slackness(x, Ac=Ac, bc=bc);
        γ;
        ψ;
        β;
        λp;
        λc;
        bp3 .- 0.1;
    ]
end

################################################################################
# problem data
################################################################################

Ap0 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
bp0 = 0.5*[
	+1,
	+1,
	+1,
	+1,
	]
friction_coefficient0 = 0.1

Ap1 = [
	+1.0 +0.0;
	+0.0 +1.0;
	-1.0 +0.0;
	+0.0 -1.0;
	]
bp1 = 0.75*[
	+1,
	+1,
	+1,
	+1,
	]
friction_coefficient1 = 0.5

# ## dynamics_model
dynamics_model = [(y, x, u) -> polytope_dynamics(y, x, u; timestep=timestep) for t = 1:horizon-1]

# ## states




file_ref = JLD2.load(joinpath(@__DIR__, "..", "examples/deps", "configuration_ref_$horizon.jld2"))
file_init = JLD2.load(joinpath(@__DIR__, "..", "examples/deps", "configuration_init_$horizon.jld2"))
configuration_ref = file_ref["configuration_ref"]
configuration_init = file_init["configuration_init"]

# ϵ_init = 0.1
# x1 = [
#     0.0; 1.0; 0.0;
#     0.0; 0.0; 0.0;
#     0.0; 0.0;
#     ϵ_init ;
#     ϵ_init ;
#     ϵ_init ;
#     ϵ_init ; ϵ_init ;
#     ϵ * ones(nh);
#     ϵ_init ;
#     vec(Ap0);
#     bp0;
# 	0.1;
#     ]
# xT = deepcopy(x1)

# ## objective
function obj_t(x, u, t)
    J = 0.0
    Δp = 1.00 .* (x[1:3] - configuration_ref[t][1:3])
    Δv = 1.00 .* (x[4:6] - configuration_ref[t][4:6])
    Δθ = 0.50 .* (x[end-3nh-1+1:end] - [vec(Ap1); bp1; friction_coefficient1])
    J += 0.5 * dot(Δp, Δp)
    J += 0.5 * dot(Δv, Δv)
    J += 0.5 * dot(Δθ, Δθ)
    return J
end

objective = [
    (x,u) -> obj_t(x, u, t) for t = 1:horizon
    ]

# ## constraints
equality_t(x, u) = contact_constraints_equality_t(x, u)
equality = [equality_t for t = 1:horizon]

inequality_t(x, u) = contact_constraints_inequality_t(x, u)
nonnegative = [inequality_t for t = 1:horizon]

# ## solver
solver = Solver(objective, dynamics_model, num_states, num_actions,
    equality=equality,
    nonnegative=nonnegative,
    options=Options()
    );

# ## initialize
state_guess = deepcopy(configuration_init)
action_guess = [zeros(0) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess)
initialize_actions!(solver, action_guess)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)
p_sol = [x[1:3] for x in x_sol]
p_truth = [x[1:3] for x in configuration_ref]

# ## visualize
Ap = mean([x[end-3nh-1+1:end-nh-1] for x in x_sol])
Ap = reshape(Ap, (nh,2))
bp = mean([x[end-nh-1+1:end-1] for x in x_sol])
friction_coefficientp = mean([x[end] for x in x_sol])

build_2d_polytope!(vis[:polytope_init], Ap1, bp1, name=:polytope, color=MeshCat.RGBA(1,1,1,0.4))
build_2d_polytope!(vis[:polytope_sol], Ap, bp, name=:polytope, color=MeshCat.RGBA(1,0,0,0.4))
build_2d_polytope!(vis[:polytope_truth], Ap0, bp0, name=:polytope, color=MeshCat.RGBA(0,0,0,1.0))
settransform!(vis[:polytope_init][:polytope], MeshCat.Translation(0.00,0,0))
settransform!(vis[:polytope_sol][:polytope], MeshCat.Translation(0.05,0,0))
settransform!(vis[:polytope_truth][:polytope], MeshCat.Translation(0.10,0,0))
anim = MeshCat.Animation()
for ii = 1:horizon+20
    atframe(anim, ii) do
		i = clamp(ii, 1, horizon)
		set_2d_polytope!(vis, p_truth[i][1:2], p_truth[i][3:3], name=:polytope_init)
		set_2d_polytope!(vis, p_sol[i][1:2], p_sol[i][3:3], name=:polytope_sol)
        set_2d_polytope!(vis, p_truth[i][1:2], p_truth[i][3:3], name=:polytope_truth)
    end
end
setanimation!(vis, anim)
u_sol
x_sol


# convert_frames_to_video_and_gif("sysid_drop_and_slide_sol")
plot(hcat(x_sol...)'[:,1:3])
plot(hcat(configuration_init...)'[:,1:3])

plot(hcat(x_sol...)'[:,4:6])
plot(hcat(configuration_init...)'[:,4:6])

plot(hcat(x_sol...)'[:,7:end])
plot(hcat(configuration_init...)'[:,7:end])
