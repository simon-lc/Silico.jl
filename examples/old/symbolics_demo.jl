using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots

vis = Visualizer()
open(vis)

include("../src/polytope.jl")
include("../src/rotate.jl")
include("../src/quaternion.jl")
include("../src/node.jl")
include("../src/body.jl")
include("../src/poly_poly.jl")
include("../src/poly_halfspace.jl")
include("../src/mechanism.jl")
include("../src/simulate.jl")
include("../src/visuals.jl")

include("../environment/polytope_bundle.jl")


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1);


mech = get_bundle_collision(; 
    timestep=0.05, 
    gravity=-9.81, 
    mass=1.0, 
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.1,
    method_type=:symbolic,
    options=Options(
        verbose=false, 
        complementarity_tolerance=1e-4,
        compressed_search_direction=false, 
        max_iterations=30,
        sparse_solver=false,
        # warm_start=true,
        )
    );


mech.solver.methods.equality_constraint
################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 1
storage = simulate!(mech, z0, H0);

solve!(mech.solver)
@benchmark $solve!($(mech.solver))

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
visualize!(vis, mech, storage)




scatter(storage.iterations)
plot!(hcat(storage.variables...)')

scatter(storage.iterations)
plot!(hcat([abs.(v)[bodies[1].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:red, legend=false)

scatter(storage.iterations)
plot!(hcat([abs.(v)[bodies[2].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:black, legend=false)

scatter(storage.iterations)
plot!(hcat([abs.(v)[contacts[1].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:green, legend=false)
    
scatter(storage.iterations)
plot!(hcat([abs.(v)[contacts[2].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:green, legend=false)
    
scatter(storage.iterations)
plot!(hcat([abs.(v)[contacts[3].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:green, legend=false)
    
scatter(storage.iterations)
plot!(hcat([abs.(v)[contacts[4].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:green, legend=false)
    
scatter(storage.iterations)
plot!(hcat([abs.(v)[contacts[5].index.variables] for v in storage.variables]...)', 
    linewidth=3, color=:green, legend=false)



plot(mech.solver.solution.all)
scatter(mech.solver.solution.all[contacts[4].index.variables])

mech.solver.solution.all[contacts[4].index.variables]





# velocity of body 1 along y
plot(hcat([abs.(s[solver.indices.primals])[2:2] for s in solutions]...)', legend=false)

plot(hcat([abs.(s[solver.indices.duals])[1:12] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[13:21] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[22:30] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[22:30][3:3] for s in solutions]...)', legend=false)

# open(vis)
# RobotVisualizer.convert_frames_to_video_and_gif("non_polytope_bundle_slow_low_friction")