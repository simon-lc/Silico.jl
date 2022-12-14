# using StaticArrays
# using Quaternions
# using GeometryBasics

using GeometryBasics
using GLVisualizer
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions


vis = Visualizer()
open(vis)
render(vis)

include("../src/Silico.jl")

include("../environment/polytope_bundle.jl")
include("../environment/polytope_drop.jl")


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1);


mech = get_sphere_bundle(;
# mech = get_polytope_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.2,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=true,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=true,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.1,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')

################################################################################
# MeshCat visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
vis, anim = visualize!(vis, mech, storage, build=true)

################################################################################
# GLVisualizer visualization
################################################################################
glvis = GLVisualizer.Visualizer()
open(glvis)

GLVisualizer.set_camera!(glvis;
    eyeposition=[5,0,0.5],
    lookat=[0,0,0.0],
    up=[0,0,1.0])
GLVisualizer.set_floor!(glvis)

build_mechanism!(glvis, mech)
for i = 1:H0
    # sleep(timestep)
    set_mechanism!(glvis, mech, storage, i)
end


################################################################################
# GLVisualizer point-cloud
################################################################################
resolution = (300, 700)
glvis = GLVisualizer.Visualizer(resolution=resolution)
open(glvis)
GLVisualizer.set_camera!(glvis;
    eyeposition=[0,-4,3.0],
    lookat=[0,0,0.0],
    up=[0,1,0.0])

build_mechanism!(glvis, mech)
GLVisualizer.set_floor!(glvis)

p1 = [Int(floor(resolution[1]/2))]
p2 = Vector(1:10:resolution[2])
n1 = length(p1)
n2 = length(p2)

depth = zeros(Float32, resolution...)
world_coordinates = zeros(3, n1 * n2)
pixel = HyperSphere(GeometryBasics.Point(0,0,0.0), 0.025)
for j = 1:n1*n2
    setobject!(vis[:pointcloud][Symbol(j)], pixel, MeshPhongMaterial(color=RGBA(0,0,0,1)))
end
for i = 1:H0
    atframe(anim, i) do
        set_mechanism!(glvis, mech, storage, i)
        depth_buffer!(depth, glvis)
        depthpixel_to_world!(world_coordinates, depth, p1, p2, glvis)
        for j in 1:n1*n2
            settransform!(vis[:pointcloud][Symbol(j)], MeshCat.Translation(world_coordinates[:,j]))
        end
    end
end
MeshCat.setanimation!(vis, anim)

linear_depth = (depth .- minimum(depth)) ./ (1e-5+(maximum(depth) - minimum(depth)))
plot(Gray.(linear_depth))


# RobotVisualizer.convert_frames_to_video_and_gif("point_cloud_sphere_bundle_tilt")
