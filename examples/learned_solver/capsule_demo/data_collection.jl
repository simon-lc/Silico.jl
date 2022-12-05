using Plots
using Statistics
using Random
using JLD2
using CUDA
using Flux
using BSON
using BenchmarkTools
CUDA.functional()

include("../methods.jl")

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
timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.9

mech = get_capsule_drop(;
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
        residual_tolerance=1e-6,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        complementarity_backstep=1e-1,
        )
    )


################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-1.0]
z0 = [xp2; vp15]
H0 = 150
u0 = zeros(3)

@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
visualize!(vis, mech, storage, build=true)

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')

mech.solver.dimensions
################################################################################
# collect simulation data
################################################################################

H_train = 250000 + 1
# H_train = 50000 + 1
Mehrotra.initialize_solver!(mech.solver)
set_input!(mech, u0)
update_parameters!(mech)
@elapsed storage_train = simulate!(mech, deepcopy(z0), H_train,
    controller=data_collection_controller)
# visualize!(vis, mech, storage_train, build=false)

H_val = 1000 + 1
Mehrotra.initialize_solver!(mech.solver)
set_input!(mech, u0)
update_parameters!(mech)
@elapsed storage_val = simulate!(mech, deepcopy(z0), H_val,
    controller=data_collection_controller)
visualize!(vis, mech, storage_val, build=true)

H_test = 5000 + 1
Mehrotra.initialize_solver!(mech.solver)
set_input!(mech, u0)
update_parameters!(mech)
@elapsed storage_test = simulate!(mech, deepcopy(z0), H_test,
    controller=data_collection_controller)
# visualize!(vis, mech, storage_test, build=false)
