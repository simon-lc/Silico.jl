using Plots
using Statistics
using Random

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanisms
################################################################################
A0 = [
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 -0.0;
        -0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.25*[+1.0, +1.0, +1.0, +1.0],
        0.25*[+1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.10
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.50

mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # complementarity_backstep=1e-1,
        )
    )

bilevel_mech = get_bilevel_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        # warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )



################################################################################
# simulation
################################################################################
H = 25

x12  = [+0.20, +0.25, -0.00]
v115 = [+0.00, +0.00, -0.00]
x22  = [+0.00, +0.75, -0.00]
v215 = [+0.00, +0.00, -0.00]
z0 = [x12; v115; x22; v215]

set_gravity!(mech, gravity)
set_gravity!(bilevel_mech, gravity)

Mehrotra.initialize_solver!(mech.solver)
Mehrotra.initialize_solver!(bilevel_mech.solver)

@elapsed storage = simulate!(mech, deepcopy(z0), H)
@elapsed bilevel_storage = simulate!(bilevel_mech, deepcopy(z0), H)

vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.3))
vis, anim = visualize!(vis, bilevel_mech, bilevel_storage, animation=anim, name=:bilevel)

scatter(storage.iterations, color=:red)
scatter!(bilevel_storage.iterations, color=:blue)

m = momentum(mech, storage)
bilevel_m = momentum(bilevel_mech, bilevel_storage)

plot(m[1,:], color="red")
plot!(m[2,:], color="green")
plot!(m[3,:], color="blue")

plot(bilevel_m[1,:], color="red")
plot!(bilevel_m[2,:], color="green")
plot!(bilevel_m[3,:], color="blue")

################################################################################
# performance evaluation
################################################################################
timesteps = 1 ./ [10, 20, 30, 50, 70, 100]
complementarity_tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
initial_conditions = [[0.02i, 0.25, 0, 0,0,0, 0, 0.75, 0, 0,0,0] for i = 1:10]


momentum_evaluation(mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)
momentum_evaluation(bilevel_mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)

horizon = 1.0
results = grid_benchmark_evaluation(
    mech,
    timesteps,
    complementarity_tolerances,
    initial_conditions,
    horizon,
    evaluation_metric=momentum_evaluation,
    vis=vis,
    verbose=true)
bilevel_results = grid_benchmark_evaluation(
    bilevel_mech,
    timesteps,
    complementarity_tolerances,
    initial_conditions,
    horizon,
    evaluation_metric=momentum_evaluation,
    vis=vis,
    verbose=true)

folder_path = joinpath(@__DIR__, "momentum_data")

plt = process_momentum(timesteps, complementarity_tolerances,
    initial_conditions, results,
    offset=[0,0],
    suffix="single_level",
    folder_path=folder_path)
plt = process_momentum(timesteps, complementarity_tolerances,
    initial_conditions, bilevel_results,
    offset=[1+length(timesteps),0],
    suffix="bilevel",
    folder_path=folder_path)

# saved_bilevel_results = deepcopy(bilevel_results)
