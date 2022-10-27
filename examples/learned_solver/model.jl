using Plots
using Statistics
using Flux
using JLD2
using CUDA
using Flux
using BSON
CUDA.functional()

include("methods.jl")

################################################################################
# load data
################################################################################
batch_size = 500
x_train, y_train, x_val, y_val, x_test, y_test, μ, σ = load_dataset(; name="dataset0")
n_input = size(x_train, 1)

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
baseline_model(x) = ((x .* σ) .+ μ)[1:10,:]
baseline_model(x_train)

cpu_model = Chain(
    Dense(n_input => 80, relu),
    Dense(80 => 50, relu),
    Dense(50 => 30, relu),
    Dense(30 => 10, sigmoid))
model = fmap(cu, cpu_model)
parameters = Flux.params(model)


################################################################################
# define losses
################################################################################
loss(x, y) = Flux.Losses.mse(model(x), y)
baseline_loss(x, y) = Flux.Losses.mse(baseline_model(x), y)

loss(x_train, y_train)
baseline_loss(x_train, y_train)


################################################################################
# training
################################################################################
n_epoch = 21
optimizer = Adam(0.001, (0.9, 0.999), 1.0e-8)
validation_loss() = round(loss(x_val, y_val), digits=4)

train_model!(train_loader, loss, parameters, optimizer, n_epoch;
    validation_loss=validation_loss,
    print_epoch=5)

save_model(model, name="model0")
loaded_cpu_model = load_model(name="model0")


################################################################################
# Analysis
################################################################################
loss(x_train, y_train)
loss(x_val, y_val)
loss(x_test, y_test)
baseline_loss(x_train, y_train)
baseline_loss(x_val, y_val)
baseline_loss(x_test, y_test)

sum(error_distribution(x_train, y_train, m=model)[1:1]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=model)[1:1]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=model)[1:1]) / size(x_test,2)

sum(error_distribution(x_train, y_train, m=model)[1:2]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=model)[1:2]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=model)[1:2]) / size(x_test,2)

sum(error_distribution(x_train, y_train, m=model)[1:3]) / size(x_train,2)
sum(error_distribution(x_val,   y_val,   m=model)[1:3]) / size(x_val,2)
sum(error_distribution(x_test,  y_test,  m=model)[1:3]) / size(x_test,2)

error_distribution(x_train, y_train, m=baseline_model)[1] / size(x_train,2)
error_distribution(x_val, y_val, m=baseline_model)[1] / size(x_val,2)
error_distribution(x_test, y_test, m=baseline_model)[1] / size(x_test,2)

binary_projection(x_train, y_train, 0.05, m=model)

error_distribution(x_test,  y_test, m=model)

######################
# false active histogram distribution
# false inactive histogram distribution
######################

# three sets
    # sure active (no mistakes allowed) -> set as equality constraints
    # unsure -> kept as is
    # sure inactive (no mistakes allowed) -> removed from optimization problem
end

# change loss function to bias towards false inactive
# actually not really

# implement active set solver
    # newton's method
    # 2 guesses of use the full NCP solver
# end
