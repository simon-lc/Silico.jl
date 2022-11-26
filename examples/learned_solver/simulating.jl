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
# test learned model
################################################################################

x_train, y_train, x_val, y_val, x_test, y_test, μ, σ = load_dataset(; name="dataset3")

cpu_model = load_model(name="model3")

error_train = error_distribution(x_train, y_train, m=cpu_model) / size(x_train, 2)
error_val = error_distribution(x_val, y_val, m=cpu_model) / size(x_val, 2)
error_test = error_distribution(x_test, y_test, m=cpu_model) / size(x_test, 2)


################################################################################
# solve
################################################################################
function newton_solve!(solver, parameters, previous_variables, mask;
        max_iterations=20,
        single_jacobian=true,
        active_regularization=1e-8,
        residual_tolerance=2solver.options.complementarity_tolerance,
        )

    indices = solver.indices
    num_cone = solver.dimensions.cone
    idx_primals = indices.primals
    idx_duals = indices.duals
    idx_slacks = indices.slacks
    mask_variables = [idx_primals; idx_duals[mask]; idx_slacks[.!mask]]
    mask_residual = [idx_primals; idx_duals]

    function local_residual(variables)
        res = residual(variables, parameters, solver)
        return res[mask_residual]
    end

    function local_residual_jacobian(variables)
        jac = residual_jacobian(variables, parameters, solver)
        num_primals = solver.dimensions.primals
        @views J = jac[mask_residual, mask_variables]
        J[num_primals+1:end, num_primals+1:end] .-=
            active_regularization * [I(num_cone)[:, mask] 0*I(num_cone)[:, .!mask]]
        return J
    end

    variables = copy(previous_variables)
    variables[idx_duals[.!mask]] .= 0.0 # inactive constraints duals = 0.0 slacks >= 0.0
    variables[idx_slacks[mask]] .= 0.0 # active constraints duals >= 0.0 slacks = 0.0
    J = local_residual_jacobian(variables)
    success = false
    for i = 1:max_iterations
        variables
        r = local_residual(variables)
        # println("iter: ", i, "   res: ", round(norm(r, Inf), digits=6))
        violation = norm(r, Inf)
        success = violation <= residual_tolerance
        success && break
        !single_jacobian && (J = local_residual_jacobian(variables))
        variables[mask_variables] -= 1.0 * (J \ r)
        # variables[idx_duals] .= max.(0.0, variables[idx_duals])
        # variables[idx_slacks] .= max.(0.0, variables[idx_slacks])
    end

    println("duals  ", round.(variables[idx_duals], digits=2))
    println("slacks ", round.(variables[idx_slacks], digits=2))

    dual_violations = variables[idx_duals] .<= -1e-5
    slack_violations = variables[idx_slacks] .<= -1e-5
    success = success && sum(dual_violations) == 0 && sum(slack_violations) == 0
    return variables, success, dual_violations, slack_violations
end

function active_set_solve!(solver, parameters, previous_variables, mask;
        max_iterations=20,
        single_jacobian=true,
        active_regularization=1e-8,
        residual_tolerance=2solver.options.complementarity_tolerance,
        )

    variables = copy(previous_variables)
    success = false
    for i = 1:10
        variables, success, dual_violations, slack_violations = newton_solve!(
            solver, parameters, variables, mask;
            max_iterations=max_iterations,
            single_jacobian=single_jacobian,
            active_regularization=active_regularization,
            residual_tolerance=residual_tolerance,
            )
        println("active iter: ", i,
            "   dual_vio: ", sum(dual_violations),
            "   slack_vio: ", sum(slack_violations),
            "    success: ", success)
        success && break
        mask = mask .&& .!dual_violations
        mask = mask .|| slack_violations
    end

    return variables, success, mask
end

i0 = 111
set_mechanism!(vis, mech, storage_test, i0)
parameters = deepcopy(storage_test.parameters[i0])
previous_variables = deepcopy(storage_test.variables[i0-1])
current_variables = deepcopy(storage_test.variables[i0])


xi_raw, yi = extract_feature_label(mech, storage_test, i0)
xi = (xi_raw .- μ) ./ (1e-5 .+ σ)
ŷi = cpu_model(xi)

scatter(yi, label="truth")
scatter!(ŷi, label="guess")
true_mask = yi
guessed_mask = Bool.(round.(ŷi, digits=0))
false_mask = deepcopy(true_mask)
false_mask[1] = 0
abs.(true_mask - guessed_mask)


# Mehrotra.initialize_solver!(mech.solver)
# opt_vars, success = newton_solve!(
#     mech.solver,
#     parameters,
#     previous_variables,
#     # current_variables,
#     # true_mask,
#     guessed_mask,
#     # false_mask,
#     single_jacobian=false,
#     max_iterations=100,
#     residual_tolerance=1e-5,
#     active_regularization=-1e-10,
#     )

Mehrotra.initialize_solver!(mech.solver)
opt_vars, success = active_set_solve!(
    mech.solver,
    parameters,
    previous_variables,
    # current_variables,
    true_mask,
    # guessed_mask,
    # false_mask,
    single_jacobian=false,
    max_iterations=100,
    residual_tolerance=1e-5,
    active_regularization=-1e-10,
    )


set_mechanism!(vis, mech, storage_test, i0)
set_mechanism!(vis, mech, [parameters[1:3]; zeros(3)])
# set_mechanism!(vis, mech, [parameters[1:3] + timestep * previous_variables[1:3]; zeros(3)])
set_mechanism!(vis, mech, [parameters[1:3] + timestep * opt_vars[1:3]; zeros(3)])
set_mechanism!(vis, mech, [parameters[1:3] + timestep * current_variables[1:3]; zeros(3)])



plot(current_variables, linewidth=3.0, label="current")
plot!(previous_variables, linewidth=3.0, label="previous")
plot!(opt_vars, linewidth=3.0, label="opt")

current_variables[1:6]
current_variables[7:16]
current_variables[17:26]
plot(1:10, log.(10, current_variables[7:16]))
plot!(1:10, log.(10, current_variables[17:26]))
opt_vars[1:6]
opt_vars[7:16]
opt_vars[17:26]

scatter(opt_vars .- current_variables, linewidth=3.0, label="opt")
scatter!(previous_variables .- current_variables, linewidth=3.0, label="previous")
plot!([6.5,6.5], [-1,1])
plot!([16.5,16.5], [-1,1])
plot!([26.5,26.5], [-1,1])


# Mehrotra.initialize_solver!(mech.solver)
# opt_vars, success = newton_solve!(
#     mech.solver,
#     parameters,
#     previous_variables,
#     guessed_mask)
#
# Mehrotra.initialize_solver!(mech.solver)
# opt_vars, success = newton_solve!(
#     mech.solver,
#     parameters,
#     previous_variables,
#     false_mask)





# @benchmark opt_vars, success = newton_solve!(
#     mech.solver,
#     parameters,
#     previous_variables,
#     guessed_mask)
#
# Main.@profiler [newton_solve!(
#     mech.solver,
#     parameters,
#     previous_variables,
#     guessed_mask) for i = 1:100000]
