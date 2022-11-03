################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, origin=[-0.05, 0, 0], x=0.1, color=RGBA(0.8,0.8,0.8,1))
set_light!(vis, direction="Negative", ambient=0.60)
set_background!(vis)
set_camera!(vis, zoom=20.0, cam_pos=[30.0, 0, 0])

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)

################################################################################
# define mechanisms
################################################################################
dir(θ) = [cos(2π * θ) sin(2π * θ)]
A=[[
    dir(0.2); dir(0.35); dir(0.6); dir(0.8); dir(0.9);],
    [dir(0.1); dir(0.4); dir(0.7);],
    ]
b=[ 0.40*[+1.0, +1.0, +2.7, +1.0, +1.0],
    0.15*[+1.0, +1.0, +1.0],
]

timestep = 0.01
gravity = -0*9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.1

mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    # method_type=:symbolic,
    method_type=:finite_difference,
    A=A,
    b=b,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        )
    )


################################################################################
# simulation
################################################################################
begin
    H = 1

    x0 = [+0.000, +2.500, +0.000]
    x1 = [+0.350, +2.965, +0.000]
    x2 = [-0.15975, +3.1824, +3.500]
    x3 = [-0.700, +2.790, +3.500]
    x4 = [-1.070, +1.980, +1.200]
    x5 = [+1.000, +2.300, -2.500]
    v0 = [-0.0, +0.0, -0.0]
    z1 = [x0; v0; x1; v0]
    z2 = [x0; v0; x2; v0]
    z3 = [x0; v0; x3; v0]
    z4 = [x0; v0; x4; v0]
    z5 = [x0; v0; x5; v0]
end

begin
    storage_1 = simulate!(mech, deepcopy(z1), H)
    storage_2 = simulate!(mech, deepcopy(z2), H)
    storage_3 = simulate!(mech, deepcopy(z3), H)
    storage_4 = simulate!(mech, deepcopy(z4), H)
    storage_5 = simulate!(mech, deepcopy(z5), H)
    vis, anim = visualize!(vis, mech, storage_1, color=green, name=:s1)
    vis, anim = visualize!(vis, mech, storage_2, color=green, animation=anim, name=:s2)
    vis, anim = visualize!(vis, mech, storage_3, color=green, animation=anim, name=:s3)
    vis, anim = visualize!(vis, mech, storage_4, color=green, animation=anim, name=:s4)
    vis, anim = visualize!(vis, mech, storage_5, color=green, animation=anim, name=:s5)
    vis, anim = visualize!(vis, mech, storage_5, color=turquoise, animation=anim, name=:s6)
end

settransform!(vis[:s4], MeshCat.Translation(1e-2,0.0,0.0))
