# using Plots
using Statistics
using Random
using JLD2
using CUDA
using Flux
using BSON
using BenchmarkTools
CUDA.functional()

include("methods.jl")

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# Test learned model on out-of-distribution data
################################################################################
timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.9

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
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        complementarity_backstep=1e-1,
        )
    )

xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]

H_ood = 5000 + 1
Mehrotra.initialize_solver!(mech.solver)
@elapsed storage_ood = simulate!(mech, deepcopy(z0), H_ood,
    controller=data_collection_controller)
visualize!(vis, mech, storage_ood, build=true)

x_ood_raw, y_ood = extract_feature_label(mech, storage_ood)
x_train, y_train, x_val, y_val, x_test, y_test, μ, σ = load_dataset(; name="dataset3")
x_ood = (x_ood_raw .- μ) ./ (1e-5 .+ σ)

@show norm(x_ood_raw)
@show norm(x_ood_raw .- μ)
# @show norm((x_ood_raw .- μ) ./ (1e-0 .+ σ))
# @show norm((x_ood_raw .- μ) ./ (1e-1 .+ σ))
# @show norm((x_ood_raw .- μ) ./ (1e-2 .+ σ))
# @show norm((x_ood_raw .- μ) ./ (1e-3 .+ σ))
# @show norm((x_ood_raw .- μ) ./ (1e-4 .+ σ))
# @show norm((x_ood_raw .- μ) ./ (1e-5 .+ σ))
@show norm(x_ood)
@show norm(x_test)
@show norm(μ)
@show norm(σ)


# idx = 33:38
# plot(mean(x_ood, dims=2)[idx]/1000)
# plot!(mean(x_ood_raw, dims=2)[idx])
# plot(std(x_ood, dims=2)[idx]/1000)
# plot!(std(x_ood_raw, dims=2)[idx])
#
# plot(mean(x_train, dims=2)[idx]/1000)
# plot!(mean(x_train_raw, dims=2)[idx])
# plot(std(x_train, dims=2)[idx]/1000)
# plot!(std(x_train_raw, dims=2)[idx])
#
# plot(log.(10, abs.(mean(x_ood_raw, dims=2))))
# plot!(log.(10, σ))

# mech.solver.dimensions


################################################################################
# test learned model
################################################################################
cpu_model = load_model(name="model3")

error_train = error_distribution(x_train, y_train, m=cpu_model) / size(x_train, 2)
error_val = error_distribution(x_val, y_val, m=cpu_model) / size(x_val, 2)
error_test = error_distribution(x_test, y_test, m=cpu_model) / size(x_test, 2)
error_ood = error_distribution(x_ood, y_ood, m=cpu_model) / size(x_ood, 2)
