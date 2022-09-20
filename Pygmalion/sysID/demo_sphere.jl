# ## dependencies
using Pkg
using CALIPSO
using LinearAlgebra
using MeshCat
using Statistics

# ## visualizer
# vis = Visualizer()
# render(vis)


# ## horizon
horizon = 21
timestep = 0.06

# ## dimensions
nx = 3 + 3 + 1 + 1 # p3, v25, γ, θ
nu = 0

num_states = [nx for t = 1:horizon]
num_actions = [nu for t = 1:horizon-1]

function sphere_dynamics(y, x, u, timestep)
    mass = 1.0
    inertia = 0.2
    gravity = -10.0
    M = [mass, mass, inertia]

    p2 = x[1:3]
    v15 = x[4:6]
    p3 = y[1:3]
    v25 = y[4:6]
    γ = y[7]
    radius_x = x[8]
    radius_y = y[8]

    p1 = p2 - timestep * v15

    res = [
        p2 + timestep * v25 - p3;
        M ./ timestep .* (p3 - 2p2 + p1) - timestep .* [0; mass .* gravity; 0] - [0; γ; 0];
        radius_x - radius_y;
    ]
    return res
end

function contact_constraints_equality_t(x, u, timestep)
    γ = x[7]
    radius = x[8]
    ϕ = x[2] - radius

    [
        γ * ϕ;
    ]
end

function contact_constraints_inequality_t(x, u, timestep)
    γ = x[7]
    radius = x[8]
    ϕ = x[2] - radius
    [
        ϕ;
        γ;
    ]
end


# ## dynamics
dynamics = [(y, x, u) -> sphere_dynamics(y, x, u, timestep) for t = 1:horizon-1]

# ## states
x1 = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.5]
xT = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.5]
x_ref = [[0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.5] for i = 1:horizon]

# ## intermediate states
xM1 = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.5]
xM2 = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.5]
tM1 = 11
tM2 = 16

# # ## objective
# function obj1(x, u)
#     J = 0.0
#     v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
#     J += 0.5 * 1.0e-1 * dot(v, v)
#     Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
#     J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
#     J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
#     return J
# end
#
# function objt(x, u)
#     J = 0.0
#     v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
#     J += 0.5 * 1.0e-1 * dot(v, v)
#     Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
#     J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
#     J += 0.5 * transpose(u) * Diagonal([1.0e-1 * ones(2); 0.1 * ones(1)]) * u
#     return J
# end
#
# function objT(x, u)
#     J = 0.0
#     v = (x[4 .+ (1:4)] - x[1:4]) ./ timestep
#     J += 0.5 * 1.0e-1 * dot(v, v)
#     Δcup_goal = x[4 .+ (1:2)] - xT[4 .+ (1:2)]
#     J += 0.5 * 1.0 * dot(Δcup_goal, Δcup_goal)
#     Δcup_ball = x[4 .+ (1:2)] - x[6 .+ (1:2)]
#     J += 0.5 * 1.0 * dot(Δcup_ball, Δcup_ball)
#     return J
# end

function obj_t(x, u, t)
    J = 0.0
    Δx = (x[1:6] - x_ref[t][1:6])
    J += 0.5 * dot(Δx, Δx)
    return J
end

objective = [
    (x,u) -> obj_t(x, u, t) for t = 1:horizon
    ]

# ## constraints
# function equality_1(x, u)
#     [
#         x - x1;
#     ]
# end
#
# function equality_t(x, u)
#     [
#         contact_constraints_equality_t(ballincup, timestep, x, u);
#     ]
# end
#
# function equality_tM1(x, u)
#     [
#         contact_constraints_equality_t(ballincup, timestep, x, u);
#         x[6 .+ (1:2)] - xM1[6 .+ (1:2)];
#     ]
# end
#
# function equality_tM2(x, u)
#     [
#         contact_constraints_equality_t(ballincup, timestep, x, u);
#         x[6 .+ (1:2)] - xM2[6 .+ (1:2)];
#     ]
# end
#
# function equality_T(x, u)
#     [
#         contact_constraints_equality_t(ballincup, timestep, x, u);
#         x[1:2] - xT[1:2];
#         x[4 .+ (1:2)] - xT[4 .+ (1:2)];
#         x[6 .+ (1:2)] - xT[6 .+ (1:2)];
#     ]
# end

function equality_t(x, u)
    contact_constraints_equality_t(x, u, timestep)
end

equality = [
        equality_t for t = 1:horizon
]

# function inequality_1(x, u)
#     [
#         contact_constraints_inequality_t(ballincup, timestep, x, u);
#     ]
# end
#
# function inequality_t(x, u)
#     [
#         contact_constraints_inequality_t(ballincup, timestep, x, u);
#     ]
# end
#
# function inequality_T(x, u)
#     [
#         contact_constraints_inequality_T(ballincup, timestep, x, u);
#     ]
# end

function inequality_t(x, u)
    contact_constraints_inequality_t(x, u, timestep);
end

nonnegative = [
    inequality_t for t = 1:horizon
]

# ## solver
solver = Solver(objective, dynamics, num_states, num_actions,
    equality=equality,
    nonnegative=nonnegative,
    options=Options()
    );

# ## initialize
x_interpolation = [
    linear_interpolation(x1, xM1, 11)...,
    linear_interpolation(xM1, xM2, 6)[2:end]...,
    linear_interpolation(xM2, xT, 6)[2:end]...]
state_guess = [x_interpolation[1], [x_interpolation[t] for t = 2:horizon]...]
action_guess = [zeros(0) for t = 1:horizon-1] # may need to run more than once to get good trajectory
initialize_states!(solver, state_guess)
initialize_actions!(solver, action_guess)

# ## solve
solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

radius = mean([x[8] for x in x_sol])
setobject!(vis[:sphere], MeshCat.HyperSphere(MeshCat.Point(0,0,0.0), radius))
anim = MeshCat.Animation()
for i = 1:horizon
    atframe(anim, i) do
        settransform!(vis[:sphere], MeshCat.Translation(0, x_sol[i][1:2]...))
    end
end
setanimation!(vis, anim)
u_sol
x_sol

# ## visualize
# visualize!(vis, ballincup,
#     x_sol,
#     Δt=timestep,
#     r_cup=0.1,
#     r_ball=0.05
# )
# q_sol = [x_sol[1][1:4], [x[4 .+ (1:4)] for x in x_sol]...]
# @load "visuals/ballincup.jld2"
# include("visuals/ball_in_cup.jl")
[contact_constraints_equality_t(x_sol[i], u_sol[i], timestep) for i = 1:horizon-1]

x_sol[2]
x_sol
