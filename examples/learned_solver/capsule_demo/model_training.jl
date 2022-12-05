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


function full_feature_extraction(mechanism, storage::TraceStorage{T,H}, i::Int) where {T,H}
    @assert i > 1
    solver = mechanism.solver
    data = solver.data
    problem = solver.problem
    indices = solver.indices
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution

    idx_duals = indices.duals
    idx_slacks = indices.slacks
    idx_equality = indices.equality
    idx_slackness = indices.slackness

    previous_variables = storage.variables[i-1]
    variables = storage.variables[i]
    previous_parameters = storage.parameters[i-1]
    parameters = storage.parameters[i]
    solution.all .= previous_variables

    # compute the residual of the previous solution under the current parameters
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=false,
        compressed=false,
        sparse_solver=false)
    Mehrotra.residual!(data, problem, indices,
        residual=true,
        jacobian_variables=true,
        jacobian_parameters=true,
        compressed=false,
        sparse_solver=false)

    unoptimized_residual = deepcopy(data.residual.all)
    # previous_jacobian = vec(deepcopy(data.jacobian_variables_compressed_dense))
    previous_jacobian = deepcopy(data.jacobian_variables_dense)

    δ_residual = previous_jacobian \ unoptimized_residual

    # previous_log_variables = log.(10, previous_variables[[idx_duals; idx_slacks]])
    previous_active_set = previous_variables[idx_duals] .>= previous_variables[idx_slacks]
    active_set = variables[idx_duals] .>= variables[idx_slacks]

    solution.all .= 0.0
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=false,
        compressed=false,
        sparse_solver=false)
    Mehrotra.residual!(data, problem, indices,
        residual=true,
        jacobian_variables=true,
        jacobian_parameters=true,
        compressed=false,
        sparse_solver=false)
    zero_residual = deepcopy(data.residual.all)
    previous_sdf = - zero_residual[idx_slackness][[1,5]]

    anticipated_sdf = - unoptimized_residual[idx_slackness][[1,5]] + previous_variables[idx_slacks][[1,5]]
    previous_pose = parameters[1:3]
    previous_velocity = parameters[4:6]
    input = parameters[7:9]
    xi = [
        # active_set;
        previous_active_set;
        δ_residual;
        previous_pose;
        previous_velocity;
        input;
        previous_sdf; # sdf(previous)
        anticipated_sdf; # sdf(previous + Δt vel)
    ]
    yi = active_set
    return xi, yi
end


mech.solver.solution.primals
mech.solver.solution.duals
mech.solver.parameters
mech.bodies[1]


################################################################################
# build dataset
################################################################################
# save_storage(storage_train, storage_val, storage_test, name="capsule_storage_0")
storage_train, storage_val, storage_test = load_storage(name="capsule_storage_0")

x_train_raw, y_train = extract_feature_label(mech, storage_train, full_feature_extraction)
x_val_raw, y_val = extract_feature_label(mech, storage_val, full_feature_extraction)
x_test_raw, y_test = extract_feature_label(mech, storage_test, full_feature_extraction)

impact_feature_extraction

μ0 = vec(mean(x_train_raw, dims=2))
σ0 = vec(std(x_train_raw .- μ0, dims=2))

x_train = (x_train_raw .- μ0) ./ (1e-5 .+ σ0)
x_val = (x_val_raw .- μ0) ./ (1e-5 .+ σ0)
x_test = (x_test_raw .- μ0) ./ (1e-5 .+ σ0)

save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, μ0, σ0, name="capsule_dataset_0")

norm(μ0)
norm(σ0)



################################################################################
# load data
################################################################################
batch_size = 500
x_train, y_train, x_val, y_val, x_test, y_test, μ, σ = load_dataset(; name="capsule_dataset_0")
n_input = size(x_train, 1)
n_output = size(y_train, 1)

train_loader = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
x_train = gpu(x_train)
y_train = gpu(y_train)
x_val = gpu(x_val)
y_val = gpu(y_val)
x_test = gpu(x_test)
y_test = gpu(y_test)
μ = gpu(μ)
σ = gpu(σ)

################################################################################
# define models
################################################################################
baseline_model(x) = ((x .* (1e-5 .+ σ0)) .+ μ0)[1:n_output,:]
baseline_model(x_train)

cpu_model = Chain(
    Dense(n_input => 80, tanh),
    Dense(80 => 50, tanh),
    Dense(50 => 30, tanh),
    Dense(30 => n_output, sigmoid))
model = fmap(cu, cpu_model)
parameters = Flux.params(model)


yh1 = [0 1.0]
y1 = [1 1.0]
Flux.Losses.binarycrossentropy(yh1, y1, ϵ=1e-5)
Flux.Losses.logitbinarycrossentropy(yh1, y1)

################################################################################
# define losses
################################################################################
loss(x, y) = Flux.Losses.mse(model(x), y) # maybe cross entropy loss
loss(x, y) = Flux.Losses.binarycrossentropy(model(x), y, ϵ=1e-5) # maybe cross entropy loss
baseline_loss(x, y) = Flux.Losses.mse(baseline_model(x), y)
baseline_loss(x, y) = Flux.Losses.binarycrossentropy(baseline_model(x), y, ϵ=1e-5)

loss(x_train, y_train)
baseline_loss(x_train, y_train)


################################################################################
# training
################################################################################
n_epoch = 31
optimizer = Adam(0.001, (0.9, 0.999), 1.0e-8)
training_loss() = round(loss(x_train, y_train), digits=4)
validation_loss() = round(loss(x_val, y_val), digits=4)

train_model!(train_loader, loss, parameters, optimizer, n_epoch;
    training_loss=training_loss,
    validation_loss=validation_loss,
    print_epoch=5)

save_model(model, name="capsule_model_0")
loaded_cpu_model = load_model(name="capsule_model_0")


################################################################################
# Analysis
################################################################################
loss(x_train, y_train)
loss(x_val, y_val)
loss(x_test, y_test)
baseline_loss(x_train, y_train)
baseline_loss(x_val, y_val)
baseline_loss(x_test, y_test)

sum(error_distribution(x_train, y_train, m=loaded_cpu_model)[1:1]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=loaded_cpu_model)[1:1]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=loaded_cpu_model)[1:1]) / size(x_test,2)

sum(error_distribution(x_train, y_train, m=loaded_cpu_model)[1:2]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=loaded_cpu_model)[1:2]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=loaded_cpu_model)[1:2]) / size(x_test,2)

sum(error_distribution(x_train, y_train, m=loaded_cpu_model)[1:3]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=loaded_cpu_model)[1:3]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=loaded_cpu_model)[1:3]) / size(x_test,2)

error_distribution(x_train, y_train, m=baseline_model)[1] / size(x_train,2)
error_distribution(x_val, y_val, m=baseline_model)[1] / size(x_val,2)
error_distribution(x_test, y_test, m=baseline_model)[1] / size(x_test,2)

error_distribution(x_test,  y_test, m=baseline_model)
error_distribution(x_test,  y_test, m=loaded_cpu_model)
binary_projection(x_train, y_train, 0.05, m=model)
