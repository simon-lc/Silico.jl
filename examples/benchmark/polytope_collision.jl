using Plots

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
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ]
b0 = [
        0.50*[+1.0, +1.0, +1.0, +1.0],
        0.35*[+1.0, +1.0, +1.0, +1.0],# +1.0],
    ]

timestep = 0.05
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
        verbose=true,
        complementarity_tolerance=1e-10,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=true,
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
        verbose=true,
        complementarity_tolerance=1e-6,
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
H = 60

################################################################################
# test no gravity
################################################################################
xp2 = [+2.00, +2.00, -0.0]
xc2 = [-0.00, +2.00, -0.5]
vp15 = [-1.0, +0.0, -0.0]
vc15 = [+1.0, -0.0, +0.0]
z0 = [xp2; vp15; xc2; vc15]


set_gravity!(mech, 0*gravity)
set_gravity!(bilevel_mech, 0*gravity)

Mehrotra.initialize_solver!(mech.solver)
Mehrotra.initialize_solver!(bilevel_mech.solver)

@elapsed storage = simulate!(mech, deepcopy(z0), H)
@elapsed bilevel_storage = simulate!(bilevel_mech, deepcopy(z0), H)

vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.3))
vis, anim = visualize!(vis, bilevel_mech, bilevel_storage, animation=anim, name=:bilevel)

scatter(storage.iterations, color=:red)
scatter!(bilevel_storage.iterations, color=:blue)

################################################################################
# test with gravity
################################################################################
xp2 = [+0.25, +0.60, -0.0]
xc2 = [-0.00, +1.50, -0.0]
vp15 = [-0.0, +0.0, -3.0]
vc15 = [+0.0, -0.0, +3.0]

z0 = [xp2; vp15; xc2; vc15]

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
