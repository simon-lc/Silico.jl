################################################################################
# visualization
################################################################################
vis = Silico.Visualizer()
Silico.open(vis)
Silico.set_floor!(vis)
Silico.set_light!(vis)
Silico.set_background!(vis)

################################################################################
# define mechanisms
################################################################################
A0 = [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        +1.0 -0.2;
        -1.0 -0.2;
    ]
b0 = 0.249*[+1.0, +3.0, +1.0, +1.0, 0.9, 0.9]

timestep = 0.05
gravity = 0.0#-9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.5

mech = Silico.get_polytope_insertion(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-5,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )


################################################################################
# simulation
################################################################################
H = 40
x2 = [+0.00, +1.75, +1.57]
v15 = [-0.0, +0.0, -0.0]
z0 = [x2; v15]

Silico.set_gravity!(mech, gravity)
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, deepcopy(z0), H)
vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.8))

storage.z[end]

# RobotVisualizer.convert_frames_to_video_and_gif("peg_in_hole_0.5percent")

################################################################################
# planning with iLQR
################################################################################
using IterativeLQR

# ## dimensions 
n = 6;
m = 3;
T = 31;

goal = [0.0, 0.25, 0.0, 0.0, 0.0, 0.0]

# test dynamics
# u0 = zeros(3)
# w0 = zeros(0)
# z1 = zero(z0)
# dz0 = zeros(length(z0), length(z0))
# du0 = zeros(length(z0), length(u0))
# Silico.dynamics(z1, mech, z0, u0)
# dynamics_jacobian_state(dz0, mech, z0, u0)
# dynamics_jacobian_input(du0, mech, z0, u0)

# ## model
dyn = IterativeLQR.Dynamics(
    (y, z, u, w) -> dynamics(y, mech, z, u),
    (dz, z, u, w) -> dynamics_jacobian_state(dz, mech, z, u),
    (du, z, u, w) -> dynamics_jacobian_input(du, mech, z, u),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout

ū = [0.1 * randn(m) for t = 1:T-1]

z̄ = IterativeLQR.rollout(model, z0, ū)
visualize!(vis, mech, z̄)

# ## objective
############################################################################
ots = [(z, u) -> transpose(z - goal) * Diagonal([0.0, 0.0, 1.0e-1, 1.0, 1.0, 1.0]) * (z - goal) + transpose(u) * Diagonal([1.0e-3, 1.0e-3, 1.0e-3]) * u for t = 1:T-1]
oT = (z, u) -> transpose(z - goal) * Diagonal(0.0 * [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * (z - goal)


cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]


# ## constraints
############################################################################

function goal_con(z, u)
    return z - goal
end

con_policyt = IterativeLQR.Constraint()
con_policyT = IterativeLQR.Constraint(goal_con, n, 0)

cons = [[con_policyt for t = 1:T-1]..., con_policyT]

# ## solver
options = IterativeLQR.Options(
        line_search=:armijo,
        max_iterations=50,
        max_dual_updates=8,
        # min_step_size=1e-2,
        objective_tolerance=1e-3,
        lagrangian_gradient_tolerance=1e-3,
        constraint_tolerance=1e-3,
        # initial_constraint_penalty=1e-1,
        scaling_penalty=10.0,
        max_penalty=1e6,
        verbose=true)

s = IterativeLQR.Solver(model, obj, cons, options=options)

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, z̄)


# ## solve
# local_callback!(solver::IterativeLQR.Solver) = continuation_callback!(solver, mech, visualize=true)
reset!(mech, residual_tolerance=1e-6, complementarity_tolerance=1e-3)
@time IterativeLQR.constrained_ilqr_solve!(s)

# ## solution
z_sol, u_sol = IterativeLQR.get_trajectory(s)

z_sol[end]
# ## visualize
z_view = [[z_sol[1] for t = 1:15]..., z_sol..., [z_sol[end] for t = 1:15]...]
visualize!(vis, mech, z_view)
