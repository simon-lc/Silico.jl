# using Pkg
# Pkg.add(path=joinpath(module_dir(), "..", "DirectTrajectoryOptimization.jl"))
using LinearAlgebra
using Plots
using DirectTrajectoryOptimization

const DTO = DirectTrajectoryOptimization

# ## horizon
T = 11

# ## particle drop
num_dynamics_state = 2
num_dynamics_parameter = 1
num_state = num_dynamics_state + num_dynamics_parameter
num_action = num_dynamics_parameter
num_parameter = 0

θ_truth = [-9.81]


function particle_explicit(x, u, w)
    z1 = x[1:num_dynamics_state]
    θ1 = x[num_dynamics_state .+ (1:num_dynamics_parameter)]
    θ2p = u[1:num_dynamics_parameter]

    m = 1.0
    timestep = 0.05
    g = θ1[1:1]

    q1 = z1[1:1]
    qd1 = z1[2:2]

    q2 = q1 + timestep .* qd1 + timestep^2/2 .* g
    qd2 = qd1 + timestep .* g

    z2 = [q2; qd2]
    return [z2; θ2p]
end

function particle_implicit(y, x, u, w)
    x2 = particle_explicit(x, u, w)
    return y .- x2
end

function rollout(x1, u, T)
    x = [x1]
    for i = 1:T-1
        push!(x, particle_explicit(x[end], u[i], zeros(num_parameter)))
    end
    return x
end


# ## model
dt = Dynamics(particle_implicit, num_state, num_state, num_action,
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1]

# ## initialization


x1 = [0.0; 0.0; -9.81]
U = [[-9.81] for i = 1:T-1]
X_truth = rollout(x1, U, T)
X = deepcopy(X_truth)
for i = 1:T
    X[i] += 0.1rand(num_state)
end

# ## objective
Q = 1.0e-2
R = 1.0e-1
Qf = 1.0e2

ot = [(x, u, w) ->
    0.5 * Q * dot(x - X[i], x - X[i]) +
    0.5 * R * dot(u - U[i], u - U[i]) +
    0.5 * R * dot(u - x[end:end], u - x[end:end]) for i=1:T-1]
oT = (x, u, w) -> 0.5 * Qf * dot(x - X[end], x - X[end])
ct = [Cost(oti, num_state, num_action,
    num_parameter=num_parameter) for oti in ot]
cT = Cost(oT, num_state, 0,
    num_parameter=num_parameter)
obj = [ct..., cT]

# ## constraints
u_bnd = 100.0
bnd1 = Bound(num_state, num_action,
    action_lower=[-u_bnd],
    action_upper=[u_bnd])
bndt = Bound(num_state, num_action,
    action_lower=[-u_bnd],
    action_upper=[u_bnd])
bndT = Bound(num_state, 0)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

# cons = [
#             Constraint((x, u, w) -> x - x1, num_state, num_action),
#             [Constraint() for t = 2:T-1]...,
#             Constraint((x, u, w) -> x - xT, num_state, 0)
#        ]
cons = [
           Constraint(),
           [Constraint() for t = 2:T-1]...,
           Constraint()
      ]

# ## problem
solver = DTO.Solver(dyn, obj, cons, bounds,
    options=DTO.Options{Float64}())

# ## initialize
u_guess = U
x_rollout = rollout(x1, U, T)

initialize_states!(solver, x_rollout)
initialize_controls!(solver, u_guess)

# ## solve
@time DTO.solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

@show x_sol[1]
@show x_sol[T]

# ## state
plot(hcat(x_sol...)')
plot(hcat(X_truth...)')

# ## control
plot(hcat(u_sol...)', linetype = :steppost)
