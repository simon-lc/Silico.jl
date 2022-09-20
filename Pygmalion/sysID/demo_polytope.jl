# ## dependencies
using Pkg
using CALIPSO
using LinearAlgebra
using MeshCat
using Statistics
using StaticArrays
using RobotVisualizer
# ## visualizer
# vis = Visualizer()
# render(vis)


# ## horizon
horizon = 11
timestep = 0.05

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
# ## dynamics
dynamics = [(y, x, u) -> polytope_dynamics(y, x, u; timestep=timestep) for t = 1:horizon-1]

# ## states
configuration_ref = [[0.0; 0.75; 0.0; 0.0; 0.0; 0.0] for i = 1:horizon]
ϵ_init = 0.1
x1 = [
    0.0; 1.0; 0.0;
    0.0; 0.0; 0.0;
    0.0; 0.0;
    ϵ_init ;
    ϵ_init ;
    ϵ_init ;
    ϵ_init ; ϵ_init ;
    ϵ * ones(nh);
    ϵ_init ;
    vec(Ap0);
    bp0;
	0.1;
    ]
xT = deepcopy(x1)

# ## objective
function obj_t(x, u, t)
    J = 0.0
    Δp = 1.00 .* (x[1:3] - configuration_ref[t][1:3])
    Δv = 1.00 .* (x[4:6] - configuration_ref[t][4:6])
    Δθ = 0.05 .* (x[end-3nh-1+1:end] - [vec(Ap0); bp0; friction_coefficient0])
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
solver = Solver(objective, dynamics, num_states, num_actions,
    equality=equality,
    nonnegative=nonnegative,
    options=Options()
    );

# ## initialize
x_interpolation = linear_interpolation(x1, xT, horizon)
state_guess = x_interpolation
action_guess = [zeros(0) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess)
initialize_actions!(solver, action_guess)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

# ## visualize
Ap = mean([x[end-3nh-1+1:end-nh-1] for x in x_sol])
Ap = reshape(Ap, (nh,2))
bp = mean([x[end-nh-1+1:end-1] for x in x_sol])
friction_coefficient = mean([x[end] for x in x_sol])
p_sol = [x[1:3] for x in x_sol]
build_2d_polytope!(vis, Ap, bp, name=:polytope)
anim = MeshCat.Animation()
for i = 1:horizon
    atframe(anim, i) do
        set_2d_polytope!(vis, x_sol[i][1:2], x_sol[i][3:3], name=:polytope)
    end
end
setanimation!(vis, anim)
u_sol
x_sol



function x_2d_rotation(q)
    # rotation from body frame to world frame
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function skew(p)
    SA[
    	 0    -p[3]  p[2];
    	 p[3]  0    -p[1];
    	-p[2]  p[1]  0;
    ]
end
