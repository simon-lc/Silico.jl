using Plots
using Statistics
using Random
using BenchmarkTools
using Flux

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
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.9,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        complementarity_backstep=1e-1,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]

u0 = zeros(3)
H0 = 150

@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
plot!(hcat(storage.variables...)')


################################################################################
# collect data
################################################################################
function ctrl(mechanism, i)
    p = mechanism.solver.solution.primals[1:3]
    u_prev = mechanism.solver.parameters[7:9]
    u = 0.9 * u_prev .+ [10, 3, 10] .* (rand(3) .- 0.5) - 1p
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    return nothing
end

H_train = 15000 + 1
@elapsed storage_train = simulate!(mech, deepcopy(z0), H_train, controller=ctrl)
# visualize!(vis, mech, storage_train, build=false)

H_val = 500 + 1
@elapsed storage_val = simulate!(mech, deepcopy(z0), H_val, controller=ctrl)
# visualize!(vis, mech, storage_val, build=false)

H_test = 5000 + 1
@elapsed storage_test = simulate!(mech, deepcopy(z0), H_test, controller=ctrl)
# visualize!(vis, mech, storage_test, build=false)

function extract_feature_label(mechanism, storage::TraceStorage{T,H}) where {T,H}
    solver = mechanism.solver

    data = solver.data
    problem = solver.problem
    indices = solver.indices
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution
    parameters = solver.parameters

    idx_duals = indices.duals
    idx_slacks = indices.slacks

    y = []
    x = []
    for i = 2:H
        previous_variables = storage.variables[i-1]
        variables = storage.variables[i]

        Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
            equality_constraint=true,
            equality_jacobian_variables=false,
            cone_constraint=false,
            cone_jacobian=false,
            sparse_solver=false,
            compressed=true)
        Mehrotra.residual!(data, problem, indices,
            residual=true,
            jacobian_variables=true,
            compressed=true,
            sparse_solver=false)

        previous_residual = deepcopy(solver.data.residual.all)
        previous_active_set = previous_variables[idx_duals] .<= previous_variables[idx_slacks]
        active_set = variables[idx_duals] .<= variables[idx_slacks]
        previous_log_variables = log.(10, previous_variables[[idx_duals; idx_slacks]])
        δ_parameters = storage.parameters[i] - storage.parameters[i-1]
        yi = [previous_active_set; previous_log_variables; previous_variables; previous_residual; δ_parameters]
        xi = active_set
        push!(y, yi)
        push!(x, xi)
    end
    y = hcat(y...)
    x = hcat(x...)
    return y, x
end
log(10, 100)
storage.parameters
x_train, y_train = extract_feature_label(mech, storage_train)
x_val, y_val = extract_feature_label(mech, storage_val)
x_test, y_test = extract_feature_label(mech, storage_test)


n_input = 114
data_train = [(x_train, y_train)]
model = Chain(
    Dense(n_input => 50, relu),
    Dense(50 => 30, relu),
    Dense(30 => 10, sigmoid))
easy_model(x) = x[1:10,:]

parameters = Flux.params(model)
loss(x, y) = Flux.Losses.mse(model(x), y, agg=mean)
easy_loss(x, y) = Flux.Losses.mse(easy_model(x), y, agg=mean)



loss(x_train, y_train)
easy_loss(x_train, y_train)

opt = Adam(0.001, (0.9, 0.999), 1.0e-8) # Gradient descent with learning rate 0.1

for epoch in 1:1000
    Flux.train!(loss, parameters, data_train, opt)
    if epoch % 10 == 0
        println(
            epoch, "   ",
            round(loss(x_train, y_train), digits=4), "   ",
            round(loss(x_val, y_val), digits=4), "   ",
        )
    end
end
loss(x_train, y_train)
loss(x_val, y_val)
loss(x_test, y_test)
easy_loss(x_train, y_train)
easy_loss(x_val, y_val)
easy_loss(x_test, y_test)
int_loss(x_train, y_train)
int_loss(x_val, y_val)
int_loss(x_test, y_test)

round(0.9, digits=0)
round(0.1, digits=0)

easy_loss(x_train, y_train)
easy_loss(.!y_train, y_train)
easy_loss(dd, y_train)

dd = [.!y_train[:,1:7500]'; y_train[:,7501:15000]']'

int_loss(x, y) = Flux.Losses.mse(Int.(round.(model(x), digits=0)), y, agg=mean)







data = mech.solver.data
problem = mech.solver.problem
indices = mech.solver.indices
methods = mech.solver.methods
cone_methods = mech.solver.cone_methods
solution = mech.solver.solution
parameters = mech.solver.parameters

Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
    equality_constraint=true,
    equality_jacobian_variables=false,
    cone_constraint=false,
    cone_jacobian=false,
    sparse_solver=false,
    compressed=true)

Mehrotra.residual!(data, problem, indices,
    residual=true,
    jacobian_variables=true,
    compressed=true,
    sparse_solver=false)
previous_residual
mech.solver.data.residual.all .= 0.0

mech.solver.data.residual


mech.solver.solution.all
mech.solver.solution.primals
mech.solver.solution.duals
mech.solver.solution.slacks

# sol = deepcopy(mech.solver.solution.all)
mech.solver.solution.all .= deepcopy(sol)
active_set = mech.solver.solution.duals .<= mech.solver.solution.slacks
mech.solver.solution.duals  .= 1e-0 * .!active_set .+ 1e-2
mech.solver.solution.slacks .= 1e-0 * active_set .+ 1e-2
# mech.solver.solution.duals .= 1e-0
# mech.solver.solution.slacks .= 1e-0

mech.solver.options.warm_start = true
mech.solver.options.complementarity_backstep = 0.0
mech.solver.options.verbose = true

Mehrotra.solve!(mech.solver)
mech.solver.trace.iterations

plot(mech.solver.solution.duals)
plot!(mech.solver.solution.slacks)
scatter!(active_set)





function predict(previous_variables, parameters, mechanism::Mechanism)
    backstep = 1e-2

    solver = mechanism.solver
    idx_duals = solver.indices.duals
    idx_slacks = solver.indices.slacks

    predicted_variables = deepcopy(previous_variables)
    predicted_variables[idx_duals] .+= backstep
    predicted_variables[idx_slacks] .+= backstep
    return predicted_variables
end

function loss(predicted_variables, parameters, mechanism; complementarity_tolerance=1e-4)
    solver = mechanism.solver
    options = solver.options
    options.warm_start = true
    options.complementarity_backstep = 0.0
    options.differentiate = false
    options.complementarity_tolerance = complementarity_tolerance
    options.residual_tolerance = complementarity_tolerance / 10

    # set mechanism.solver.parameters and updates the nodes accordingly
    mechanism.solver.parameters .= parameters
    update_nodes!(mechanism)

    # warm-start the solver
    solver.solution.all .= predicted_variables

    # solve
    Mehrotra.solve!(solver)

    iterations = solver.trace.iterations
    l = iterations
    return l
end

x_predicted = ones(mech.dimensions.variables)
θ = mech.solver.parameters
@benchmark loss(x_predicted, θ, mech)



features => fcl => relu => fcl => sigmoid <> label
