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
# define mechanism
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 0.2;
inertia = 0.8 * Matrix(Diagonal(ones(3)));
inertia = Matrix(Diagonal([0.1, 0.5, 0.9]));
friction_coefficient = 0.5

mech = get_3d_sphere_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
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
    )

################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +0.00, +1.00, 1,0,0,0]
vp15 = [+0.00, -0.00, +0.00, +0.0,+5.0,+0.0]

norm(50*ones(3) * timestep)

z0 = [xp2; vp15]

H0 = 100
@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)

scatter(storage.iterations .== 30)
scatter(storage.iterations)
plot!(hcat([storage.v[i][1][4:6] for i=1:H0]...)')
# plot(hcat(storage.variables...)')
# plot!(hcat([storage.v[i][1][4:6] for i=1:H0]...)')

# plot(mech.solver.data.residual.all)

# x0 = 0.5 * ones(3)
# x1 = 0.5 * ones(3)
# f(x) = sqrt(1 - x'*x) * Diagonal([1, 0.2, 1.0]) * x
# inert = 0.1*rand(3,3)
# inert = inert' * inert
# f(x) = sqrt(1 - x'*x) * inert * x - cross(x, inert * x)
# g(x) = sqrt(1 - x'*x) * inert * x + cross(x, inert * x)
#
# plot(hcat([f(x * ones(3)) for x ∈ 0.01:0.01:0.579]...)')
# plot!(hcat([g(x * ones(3)) for x ∈ 0.01:0.01:0.579]...)')
#
# f(0.57*ones(3))
setobject!(vis[:robot][:bodies][:pbody][:x], HyperRectangle(Vec(0,0,0), Vec(0.9, 0.1, 0.1)))
setobject!(vis[:robot][:bodies][:pbody][:y], HyperRectangle(Vec(0,0,0), Vec(0.1, 0.7, 0.1)))
setobject!(vis[:robot][:bodies][:pbody][:z], HyperRectangle(Vec(0,0,0), Vec(0.1, 0.1, 0.5)))
