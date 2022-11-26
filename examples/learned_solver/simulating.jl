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
@elapsed storage_ood = simulate!(mech, deepcopy(z0), H_ood,
    controller=data_collection_controller)
visualize!(vis, mech, storage_ood, build=true)

x_ood_raw, y_ood = extract_feature_label(mech, storage_ood)
x_train, y_train, x_val, y_val, x_test, y_test, μ, σ = load_dataset(; name="dataset2")
x_ood = (x_ood_raw .- μ) ./ (1e-5 .+ σ)
@show norm(x_ood_raw)
@show norm(x_ood)
@show norm(μ)
@show norm(σ)

################################################################################
# test learned model
################################################################################
cpu_model = load_model(name="model2")

error_train = error_distribution(x_train, y_train, m=cpu_model) / size(x_train, 2)
error_val = error_distribution(x_val, y_val, m=cpu_model) / size(x_val, 2)
error_test = error_distribution(x_test, y_test, m=cpu_model) / size(x_test, 2)
error_ood = error_distribution(x_ood, y_ood, m=cpu_model) / size(x_ood, 2)
# error_ood = error_distribution(x_ood, y_ood, m=baseline_model) / size(x_ood, 2)


################################################################################
# solve
################################################################################
mech.solver.options.verbose = true
mech.solver.options.warm_start = true
mech.solver.options.complementarity_backstep = 0

i0 = 250
mech.solver.parameters .= storage_ood.parameters[i0]
mech.solver.solution.all .= storage_ood.variables[i0-1]
Mehrotra.solve!(mech.solver)


xi_raw, yi = extract_feature_label(mech, storage_ood, i0)
xi = (xi_raw .- μ) ./ (1e-5 .+ σ)
ŷi = cpu_model(xi)
scatter(yi)
scatter!(ŷi)


function newton_solve!(solver, parameters, previous_variables, mask)
    indices = solver.indices
    idx_primals = indices.primals
    idx_duals = indices.duals
    idx_slacks = indices.slacks
    mask_variables = [idx_primals; idx_duals[mask]]

    function local_residual(variables)
        res = residual(variables, parameters, solver)
        # r_primals = res[idx_primals]
        # r_active_duals = res[idx_duals][mask] - variables[idx_slacks][mask] # sγ - ϕ - sγ
        # return [r_primals; r_active_duals]
        return res[mask_variables]
    end

    function local_residual_jacobian(variables)
        Jac = residual_jacobian(variables, parameters, solver)
        num_primals = solver.dimensions.primals
        num_active_duals = sum(mask)
        @views J = Jac[mask_variables, mask_variables]
        J[num_primals+1:end, num_primals+1:end] .-= 1e-10 * I(num_active_duals)
        return J
    end

    variables = previous_variables
    variables[idx_slacks] .= 0.0
    J0 = local_residual_jacobian(variables)
    success = false
    for i = 1:10
        variables
        r = local_residual(variables)
        violation = norm(r, Inf)
        success = violation <= 2 * solver.options.complementarity_tolerance
        success && break
        # @show norm(r, Inf)
        # plt = plot()
        # scatter!(r)
        # display(plt)
        # J = local_residual_jacobian(variables)
        # variables[mask_variables] -= J \ r
        variables[mask_variables] -= J0 \ r
    end
    return variables, success
end

mech.solver.dimensions

i0 = 1100
parameters = deepcopy(storage_ood.parameters[i0])
previous_variables = deepcopy(storage_ood.variables[i0-1])
current_variables = deepcopy(storage_ood.variables[i0])

xi_raw, yi = extract_feature_label(mech, storage_ood, i0)
xi = (xi_raw .- μ) ./ (1e-5 .+ σ)
ŷi = cpu_model(xi)

scatter(yi)
scatter!(ŷi)
true_mask = yi
guessed_mask = Bool.(round.(ŷi, digits=0))

@benchmark opt_vars, success = newton_solve!(
    mech.solver,
    parameters,
    previous_variables,
    guessed_mask)


Main.@profiler [newton_solve!(
    mech.solver,
    parameters,
    previous_variables,
    guessed_mask) for i = 1:100000]

residual(opt_vars, parameters, solver)
