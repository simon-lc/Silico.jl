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
timestep = 0.01;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.3;

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-5,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2  = [+0.0, +0.70, +0.3]
vp15 = [-0.0, -1.5, -0.0]
z0 = [xp2; vp15]

u0 = zeros(3)
H0 = 3

@elapsed storage = simulate!(mech, deepcopy(z0), H0)
mech.solver.solution

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)


Mehrotra.initialize_solver!(mech.solver)
mech.solver.options.max_iterations = 30
mech.solver.options.complementarity_tolerance = 1e-1
mech.solver.solution.primals

z1 = step!(mech, deepcopy(z0))
mech.solver.solution.primals

mech.solver.data.step.primals[1:2]
function get_pair(x0)
    Mehrotra.initialize_solver!(mech.solver)
    mech.solver.options.max_iterations = 1
    mech.solver.solution.all .= storage.variables[1]
    mech.solver.solution.all .= ones(26)
    # mech.solver.solution.all .= rand(26)
    mech.solver.solution.all[1:2] = x0

    z1 = step!(mech, deepcopy(z0))
    # mech.solver.data.step.primals[1:2]
    x1 = deepcopy(mech.solver.solution.all[1:2])
    @show x1
    return x1
end

plt = plot(legend=false)
N = 30
for i = 1:N
    for j = 1:N
        x0 = [50(i-N/2)/N, 50(j-N/2)/N]
        x1 = get_pair(x0)
        plot!(plt, [x0[1], x1[1]], [x0[2], x1[2]])
    end
end
display(plt)


storage.equality_violations
storage.cone_product_violations




# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
