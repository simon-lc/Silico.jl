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

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)
black = RGBA(0.2,0.2,0.25,1)

################################################################################
# define mechanism
################################################################################
timestep = 0.01;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.6

mech = get_diverse_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    minkowski_radius=0.3,
    union_radius=0.15,
    segment=4.0,
    A = [[0 1; +2.5 -1; -2.5 -1], [5 1; -5 1; 1 5; 1 -5.0]],
    b = [0.3 * ones(3), 0.3 * ones(4)],
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        complementarity_backstep=1e-2,
        )
    );

################################################################################
# test simulation
################################################################################
x1 = [+0.0,1.5,-0.20]
v1 = [-0,0,-1.0]
x2 = [-1.8,3.5,+0.75]
v2 = [-0,0,-0.0]
z0 = [x1; v1; x2; v2]



H0 = 250
mech.contacts
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
vis, anim = visualize!(vis, mech, storage, name=:green, color=green)
vis, anim = visualize!(vis, mech, storage, name=:turquoise, color=turquoise, animation=anim)

scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
set_floor!(vis, x=0.3)

set_camera!(vis, zoom=10.0, cam_pos=[-33, 0, 0])

################################################################################
# union and minkowski
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, origin=[0,0,-100])
set_light!(vis, ambient=1.22)
set_background!(vis)
set_camera!(vis, zoom=10.0, cam_pos=[-33, 0, 0])

minkowski_radius = 0.3
union_radius = 0.15
segment = 4.0
A = [[0 1; +2.5 -1; -2.5 -1], [5 1; -5 1; 1 5; 1 -5.0]]
normalize_A!.(A)
b = [0.3 * ones(3), 0.3 * ones(4)]

minkowski_sphere = SphereShape(minkowski_radius, [-0,0.0])
minkowski_polytope = PolytopeShape(A[2], b[2] + A[2] * [+1.2, 0])
minkowski_sum = PaddedPolytopeShape110(minkowski_radius, A[2], b[2] + A[2] * [-1.5, 0])

scale = 3
union_capsule = CapsuleShape(union_radius/scale, segment/scale)
union_polytope = PolytopeShape(A[1], b[1]/scale)

build_shape!(vis[:minkowski_sphere], minkowski_sphere, collider_color=turquoise)
build_shape!(vis[:minkowski_polytope], minkowski_polytope, collider_color=turquoise)
build_shape!(vis[:minkowski_sum], minkowski_sum, collider_color=turquoise)

build_shape!(vis[:union_capsule], union_capsule, collider_color=green)
build_shape!(vis[:union_polytope], union_polytope, collider_color=green)
build_shape!(vis[:union][:capsule], union_capsule, collider_color=green)
build_shape!(vis[:union][:polytope], union_polytope, collider_color=green)
q = RotX(-0.30π)
settransform!(vis[:union_capsule], MeshCat.compose(
    MeshCat.Translation(SVector{3}([0,+1.4,0])),
    MeshCat.LinearMap(rotationmatrix(q)),
    )
)
settransform!(vis[:union_polytope], MeshCat.compose(
    MeshCat.Translation(SVector{3}([0,+0,0])),
    MeshCat.LinearMap(rotationmatrix(q)),
    )
)
settransform!(vis[:union][:capsule], MeshCat.compose(
    MeshCat.Translation(SVector{3}([0,-1.6,0])),
    MeshCat.LinearMap(rotationmatrix(q)),
    )
)
settransform!(vis[:union][:polytope], MeshCat.compose(
    MeshCat.Translation(SVector{3}([0,-1.6,0])),
    MeshCat.LinearMap(rotationmatrix(q)),
    )
)
