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
timestep = 0.04
friction_coefficient = 0.1
mass = 1.0
mech = get_bundle_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=0.2 * ones(1,1),
    friction_coefficient=friction_coefficient,
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

solve!(mech.solver)

xp2_goal = [
    0.15-0.2978711928307936,
    # 0.5+0.6803779346453341,
    +0.6803779346453341,
    -0.1833009704915061]
xc2_goal = [
    0.15-0.09534381468491418,
    +0.16439703590459195,
    -1.7359408741345008]
xp2_start = [
    -0.2978711928307936,
    # +0.6803779346453341,
    0.5+0.6803779346453341,
    -0.1833009704915061]
xc2_start = [
    -0.09534381468491418,
    +0.16439703590459195,
    -1.7359408741345008]
xp2 = [+0.0,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]
u0 = zeros(12)
w0 = zeros(0)

# ## dimensions
n = mech.dimensions.state
m = mech.dimensions.input
nu_infeasible = 3

################################################################################
# ## simulation test
################################################################################

u_hover = [0; -0*9.81; 0; 0; 0; 0]
function ctrl!(m, i; u=u_hover)
    set_input!(m, u)
end

H0 = 50
storage = simulate!(mech, z0, H0, controller=ctrl!)
scatter(storage.iterations)
visualize!(vis, mech, storage, build=true)

################################################################################
# ## reference trajectory
################################################################################
zref = [[xp2_goal; vp15; xc2_goal; vc15] for i=1:20]
# ## horizon
T = length(zref)

set_mechanism!(vis, mech, [xp2_start; vp15; xc2_start; vc15])
set_mechanism!(vis, mech, [xp2_goal; vp15; xc2_goal; vc15])

################################################################################
# ## ILQR problem
################################################################################
# ## model
dyn = IterativeLQR.Dynamics(
    (y, z, u, w) -> dynamics(y, mech, z, u, w),
    (dz, z, u, w) -> dynamics_jacobian_state(dz, mech, z, u, w),
    (du, z, u, w) -> dynamics_jacobian_input(du, mech, z, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
z1 = [xp2_start; vp15; xc2_start; vc15]
ū = [u_hover for t = 1:T-1]

z̄ = IterativeLQR.rollout(model, z1, ū)
visualize!(vis, mech, z̄)

# ## objective
############################################################################
qt = [1e+0*ones(3); 1e+1ones(3); 1e+0*ones(3); 1e+1ones(3)]
rt = [1e-2*[1,1e-1,1,1,1e-1,1];]
ots = [(z, u) -> transpose(z - zref[t]) * Diagonal(timestep * qt) * (z - zref[t]) +
    transpose(u) * Diagonal(timestep * rt) * u for t = 1:T-1]
oT = (z, u) -> transpose(z - zref[end]) * Diagonal(timestep * qt) * (z - zref[end])


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
        1e-2 * (ul - u[3 .+ (1:nu_infeasible)]);
        1e-2 * (u[3 .+ (1:nu_infeasible)] - uu);
    ]
end

function goal(z, u)
    Δ = 1e-1 * (z - zref[end])
    return Δ
end

con_policyt = IterativeLQR.Constraint(contt, n, m, indices_inequality=collect(1:2nu_infeasible))
con_policyT = IterativeLQR.Constraint(goal, n, 0)

cons = [[con_policyt for t = 1:T-1]..., con_policyT]

# ## solver
options = IterativeLQR.Options(
        line_search=:armijo,
        max_iterations=50,
        max_dual_updates=12,
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
reset!(mech, residual_tolerance=1e-6, complementarity_tolerance=1e-1)
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


# RobotVisualizer.convert_frames_to_video_and_gif("polytope_bundle_nonconvex")
