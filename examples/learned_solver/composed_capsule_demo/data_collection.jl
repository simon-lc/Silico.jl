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


################################################################################
# build dataset
################################################################################
x_train_raw, y_train = extract_feature_label(mech, storage_train)
x_val_raw, y_val = extract_feature_label(mech, storage_val)
x_test_raw, y_test = extract_feature_label(mech, storage_test)

μ = vec(mean(x_train_raw, dims=2))
σ = vec(std(x_train_raw .- μ, dims=2))

x_train = (x_train_raw .- μ) ./ (1e-5 .+ σ)
x_val = (x_val_raw .- μ) ./ (1e-5 .+ σ)
x_test = (x_test_raw .- μ) ./ (1e-5 .+ σ)

save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, μ, σ, name="composed_capsule_dataset_0")

norm(μ)
norm(σ)

# plot(x_train[:,1:100], legend=false)
# plot(log.(1e-5 .+ abs.(x_train[:,1:1000])), legend=false)
# plot(x_train[:,end-100:end], legend=false)
# plot(μ)
# plot(σ)

N = 100
x_train[:,1:N]
x_val[:,1:N]
x_test[:,1:N]
plt = plot()
v_train = vec(mean(abs.(x_train[:,1:N]), dims=1))
v_val = vec(mean(abs.(x_val[:,1:N]), dims=1))
v_test = vec(mean(abs.(x_test[:,1:N]), dims=1))
plot!(plt, 1:N, v_train)
plot!(plt, 1:N, v_val)
plot!(plt, 1:N, v_test)
# plot!(plt, 1:N, v_ood)
