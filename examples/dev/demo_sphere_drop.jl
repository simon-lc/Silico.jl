using Plots
using Statistics
using Random
using RobotVisualizer
using MeshCat

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanism
################################################################################
timestep = 0.01;
gravity = -9.81;
mass = 1.0;
inertia = 0.05 * ones(1);

mech = get_sphere_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.3,
    # method_type=:symbolic,
    method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [-0.2,0.4,-0.30]
vp15 = [+2,4,-3.25]
z0 = [xp2; vp15]

u0 = zeros(3)
H0 = 110

@elapsed storage = simulate!(mech, z0, H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 10)

visualize!(vis, mech, storage, build=false)

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')


RobotVisualizer.convert_frames_to_video_and_gif("sphere_rotate")
RobotVisualizer.convert_frames_to_video_and_gif("gjk_long", framerate=1, width=2500)

function RobotVisualizer.convert_frames_to_video_and_gif(filename, overwrite::Bool=true;
        width=1080, framerate=5)
    RobotVisualizer.MeshCat.convert_frames_to_video(
        homedir() * "/Downloads/$filename.tar",
        homedir() * "/Documents/video/$filename.mp4",
            overwrite=overwrite,
            framerate=framerate)

    convert_video_to_gif(
        homedir() * "/Documents/video/$filename.mp4",
        homedir() * "/Documents/video/$filename.gif",
            overwrite=overwrite,
            width=width)
    return nothing
end
