using GeometryBasics
using GLVisualizer
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim


vis = Visualizer()
open(vis)

include("../src/DojoLight.jl")

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        # verbose=true,
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.1, +1.5, -0.25]
vp15 = [-0, -0, -0.0]
z0 = [xp2; vp15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)
z_final = deepcopy(storage.z[end])
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
# GLVisualizer point-cloud
################################################################################
resolution = (301, 700)
glvis = GLVisualizer.Visualizer(resolution=resolution)
open(glvis)
GLVisualizer.set_camera!(glvis;
    eyeposition=[0,+2,3.0],
    lookat=[0,0,0.0],
    up=[0,1,0.0])

build_mechanism!(glvis, mech)
GLVisualizer.set_floor!(glvis)

p1 = [Int(floor((resolution[1]+1)/2))] # perfectly centered for uneven number of pixels
p2 = Vector(1:10:resolution[2])
n1 = length(p1)
n2 = length(p2)

depth = zeros(Float32, resolution...)
world_coordinates = zeros(Float32, 3, n1 * n2)
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
    # plt = Plots.scatter(world_coordinates[2,:], world_coordinates[3,:], aspectratio=1.0)
    # display(plt)
end
MeshCat.setanimation!(vis, anim)

include("../cvxnet/softmax.jl")
include("../cvxnet/loss.jl")
include("../cvxnet/point_cloud.jl")

A0 = [
    +1.0 -0.2;
    +0.0 +1.0;
    -1.0 -0.2;
    +0.0 -1.0;
    ] .- 0.00ones(4,2);
# A0 = [
#     +1.0 -0.0;
#     +0.0 +1.0;
#     -1.0 -0.0;
#     +0.0 -1.0;
#     ] .- 0.00ones(4,2);
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1,
    +1,
    +1,
    +0,
    ];


eyeposition = [0,0,3.0]
lookat = [0,0,0.0]
up = [0,1,0.0]
E = [[0, i, 3.0] for i = -2:2]
z_nominal = zeros(6)
set_mechanism!(vis, mech, z_nominal)
WC = [point_cloud(glvis, mech, z_nominal, p1, p2, resolution, e, lookat, up) for e in E]


θ0 = pack_halfspaces(A0, b0) + 1e-0*rand(8)
δ0 = 1e+2
plot_polyhedron(A0, b0, δ0)
@time loss(WC, E, θ0, β=1e-3, Δ=0.01, δ=δ0)

function solution_trace_callback(iterate, trace)
    # @show fieldnames(typeof(iterate))
    # @show iterate
    push!(trace, iterate)
    return false
end


local_loss(θ) = loss(WC, E, θ, β=1e-3, Δ=1e-2, δ=δ0)
trace0 = []
local_callback(iterate) = solution_trace_callback(iterate, trace0)

n = 12
θ1 = [range(-π, π, length=n+1)[1:end-1]; 0.2*ones(n)]# + 1e-1rand(2n)
A1, b1 = unpack_halfspaces(θ1)
res = Optim.optimize(local_loss, θ1,
    # BFGS(),
    Newton(),
    Optim.Options(
        # f_tol=1e-3,
        allow_f_increases=true,
        # callback=local_callback,
        store_trace = false,
        show_trace = true),
    )



################################################################################
# unpack results
################################################################################
θopt = Optim.minimizer(res)
Aopt, bopt = unpack_halfspaces(θopt)
# solution_trace = [iterate.value for iterate in iterate_trace]
# iterate_trace = Optim.trace(res)


################################################################################
# visualize results
################################################################################
build_2d_polytope!(vis, A0, b0, name=:reference,
    color=RGBA(0.2,0.2,0.8,1.0))
build_2d_polytope!(vis, A1, b1, name=:initial,
        color=RGBA(1,1,0.0,0.5))
build_2d_polytope!(vis, Aopt, bopt, name=:optimized,
        color=RGBA(1,1,0.0,0.5))
settransform!(vis[:reference], MeshCat.Translation(+0.0, 0,0))
settransform!(vis[:initial], MeshCat.Translation(+0.1, 0,0))
settransform!(vis[:optimized], MeshCat.Translation(+0.2, 0,0))

# vis = Visualizer()
# open(vis)
set_floor!(vis, x=1.0, origin=[-0.5,0,0])
set_light!(vis, direction="Negative", ambient=0.5)
set_background!(vis)
