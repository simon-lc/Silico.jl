# ## setup
using Mehrotra

include("../src/DojoLight.jl")

# ## visualizer
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# ## system
################################################################################
gravity = -9.81
timestep = 0.05
friction_coefficient = 0.3
finger_friction_coefficient = 0.9
mass = 1.0
mech = get_quasistatic_manipulation(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=0.2 * ones(1,1),
    friction_coefficient=friction_coefficient,
    finger_friction_coefficient=finger_friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );

# solve!(mech.solver)

x_object_start  = [+0.00, +0.30, +0.25]
x_finger1_start = [-0.50, +0.20, -0.00]
x_finger2_start = [+0.60, +0.20, -0.00]
z_start = [x_object_start; x_finger1_start; x_finger2_start]
z0 = [x_object_start; x_finger1_start; x_finger2_start]
# x_object_goal  = [
#     0.00,
#     1.50,
#     0.25
#     ]
# x_finger1_goal = [
#     -0.40,
#     +1.40,
#     -0.00
#     ]
# x_finger2_goal = [
#     +0.55,
#     +1.40,
#     -0.00
#     ]
x_object_goal  = [
    -0.006236093,
    +1.511882639,
    +0.246224411,
    ]
x_finger1_goal = [
    -0.420458253,
    +1.389759286,
    +3.279881150e-5,
    ]
x_finger2_goal = [
    +0.576612649,
    +1.398546105,
    -0.000559764,
    ]

z_goal = [x_object_goal; x_finger1_goal; x_finger2_goal]

u0 = zeros(9)
w0 = zeros(0)

# ## dimensions
n = mech.dimensions.state
m = mech.dimensions.input
nu_infeasible = 3

################################################################################
# ## simulation test
################################################################################

u_hover = zeros(9)
function ctrl!(m, i; u=u_hover)
    set_input!(m, u)
end

H0 = 50
storage = simulate!(mech, z0, H0, controller=ctrl!)
scatter(storage.iterations)
visualize!(vis, mech, storage, build=true)

storage.x[end]
################################################################################
# ## reference trajectory
################################################################################
A = range(0, 1, length=H0)
zref = [α * z_goal  + (1 - α) * z_start for α in A]
uref = [[0; -gravity; 0; (α * z_goal  + (1 - α) * z_start)[4:9]] for α in A[1:end-1]]
# ## horizon
T = length(zref)

set_mechanism!(vis, mech, z_start)
set_mechanism!(vis, mech, z_goal)

################################################################################
# ## ILQR problem
################################################################################
# ## model
dyn = IterativeLQR.Dynamics(
    (y, z, u, w) -> dynamics(y, mech, z, u, w),
    (dz, z, u, w) -> quasistatic_dynamics_jacobian_state(dz, mech, z, u, w),
    (du, z, u, w) -> quasistatic_dynamics_jacobian_input(du, mech, z, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
z1 = deepcopy(z_start)
ū = [u_hover for t = 1:T-1]
ū = deepcopy(uref)

z̄ = IterativeLQR.rollout(model, z1, ū)
visualize!(vis, mech, z̄)

# ## objective
############################################################################
qt = [1e+0*ones(3); [10,10,1e-1]; [10,10,1e-1]]
rt = [1e-1*[1,1,1,1,1e-1,1,1,1e-1,1];]
ots = [(z, u) -> transpose(z - zref[t]) * Diagonal(timestep * qt) * (z - zref[t]) +
    transpose(u - uref[t]) * Diagonal(timestep * rt) * (u - uref[t]) for t = 1:T-1]
oT = (z, u) -> transpose(z - zref[end]) * Diagonal(timestep * 10*qt) * (z - zref[end])


cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]


# ## constraints
############################################################################
ul = -1.0 * 1e-3*ones(nu_infeasible)
uu = +1.0 * 1e-3*ones(nu_infeasible)

function contt(z, u)
    [
        # 1e-1 * (ul - u[1:nu_infeasible]);
        # 1e-1 * (u[1:nu_infeasible] - uu);
        # 1e-2 * (ul - u[3 .+ (1:nu_infeasible)]);
        # 1e-2 * (u[3 .+ (1:nu_infeasible)] - uu);
        1e-1 * (ul - u[1:nu_infeasible]);
        1e-1 * (u[1:nu_infeasible] - uu);
    ]
end

function goal(z, u)
    Δ = 1e-0 * (z - zref[end])
    return Δ
end

con_policyt = IterativeLQR.Constraint(contt, n, m, indices_inequality=collect(1:2nu_infeasible))
con_policyT = IterativeLQR.Constraint(goal, n, 0)

cons = [[con_policyt for t = 1:T-1]..., con_policyT]

# ## solver
options = IterativeLQR.Options(
        line_search=:armijo,
        max_iterations=50,
        max_dual_updates=3,
        # min_step_size=1e-2,
        objective_tolerance=1e-3,
        lagrangian_gradient_tolerance=1e-3,
        constraint_tolerance=1e-4,
        # initial_constraint_penalty=1e-1,
        scaling_penalty=10.0,
        max_penalty=1e4,
        verbose=true)

s = IterativeLQR.Solver(model, obj, cons, options=options)

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, z̄)


# ## solve
local_callback!(solver::IterativeLQR.Solver) = continuation_callback!(solver, mech, visualize=true)
reset!(mech, residual_tolerance=1e-6, complementarity_tolerance=1e-4)
@time IterativeLQR.constrained_ilqr_solve!(s, augmented_lagrangian_callback! = local_callback!)

# ## solution
z_sol, u_sol = IterativeLQR.get_trajectory(s)

# ## visualize
z_view = [[z_sol[1] for t = 1:15]..., z_sol..., [z_sol[end] for t = 1:15]...]
visualize!(vis, mech, z_view)

plot(hcat(z_sol...)')
plot(hcat(u_sol...)'[:,1:1])
plot!(hcat(u_sol...)'[:,2:2])
plot!(hcat(u_sol...)'[:,3:3])
plot!(hcat(u_sol...)'[:,4:4])
plot!(hcat(u_sol...)'[:,5:5])
plot!(hcat(u_sol...)'[:,6:6])


# RobotVisualizer.convert_frames_to_video_and_gif("ilqr_manipulation")

build_mechanism!(vis, mech, name=:reference, color=RGBA(1,0.7,0.0,1))
set_mechanism!(vis, mech, z_goal, name=:reference)
