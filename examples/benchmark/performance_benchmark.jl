using Plots
using Statistics
using Random

include("benchmark_methods.jl")

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)
set_floor!(vis, origin=[-0.05, 0, 0], x=0.1)
set_light!(vis, direction="Negative")
set_background!(vis)

################################################################################
# define mechanisms
################################################################################
A0 = [
    [
        +1.0 +0.2;
        -0.0 +1.0;
        -1.0 -0.3;
        +0.0 -1.0;
        +0.8 -0.8;
        ],
    ]
b0 = [
        0.40*[+1.0, +1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.02
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.20

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
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # warm_start=true,
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
        warm_start=false,
        # complementarity_backstep=1e-1,
        )
    )



################################################################################
# simulation
################################################################################
H = 40

xp2  = [+0.00, +0.90, -0.00]
vp15 = [-0.00, +0.00, -3.00]
z0 = [xp2; vp15]

set_gravity!(mech, gravity)
set_gravity!(bilevel_mech, gravity)

Mehrotra.initialize_solver!(mech.solver)
Mehrotra.initialize_solver!(bilevel_mech.solver)

@elapsed storage = simulate!(mech, deepcopy(z0), H)
@elapsed bilevel_storage = simulate!(bilevel_mech, deepcopy(z0), H)

vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(4/255,191/255,173/255,1.0))
vis, anim = visualize!(vis, bilevel_mech, bilevel_storage, animation=anim, name=:bilevel)

scatter(storage.iterations, color=:red)
scatter!(bilevel_storage.iterations, color=:blue)

################################################################################
# performance evaluation
################################################################################
timesteps = 1 ./ [10, 20, 30, 50, 70, 100]
complementarity_tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
z_min = [0.0, +sqrt(2)/2, 0, -1, -2, -1]
z_max = [0.0, +1.5, 2π, +1, +0, +1]
initial_conditions = generate_initial_conditions(100, z_min, z_max)


performance_evaluation(mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)
performance_evaluation(bilevel_mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)

horizon = 1.5
performances = grid_benchmark_evaluation(
    mech,
    timesteps,
    complementarity_tolerances,
    initial_conditions,
    horizon,
    evaluation_metric=performance_evaluation,
    vis=vis,
    verbose=true)
bilevel_performances = grid_benchmark_evaluation(
    bilevel_mech,
    timesteps,
    complementarity_tolerances,
    initial_conditions,
    horizon,
    evaluation_metric=performance_evaluation,
    vis=vis,
    verbose=true)



folder_path = joinpath(@__DIR__, "performance_data")

plt = process_performances(timesteps, complementarity_tolerances,
    initial_conditions, performances,
    offset=[0,0],
    suffix="single_level",
    folder_path=folder_path)
plt = process_performances(timesteps, complementarity_tolerances,
    initial_conditions, bilevel_performances,
    offset=[1+length(timesteps),0],
    suffix="bilevel",
    folder_path=folder_path)
