using Plots
using Statistics
using Random
using BenchmarkTools
using Flux
using CUDA
using BSON
CUDA.functional()





data_train = [(x_train_norm, y_train)]
batch_size = 500
train_loader = Flux.DataLoader((x_train_norm, y_train), batchsize=batch_size, shuffle=true)


easy_model(x) = x[1:10,:]
model = Chain(
    Dense(n_input => 80, relu),
    Dense(80 => 50, relu),
    Dense(50 => 30, relu),
    Dense(30 => 10, sigmoid))
model = fmap(cu, model)
parameters = Flux.params(model)
parameters.params

loss(x, y) = Flux.Losses.mse(model(x), y)
easy_loss(x, y) = Flux.Losses.mse(easy_model(x), y)
int_loss(x, y) = Flux.Losses.mse(round.(model(x), digits=0), y)

function hist_dist(x, y; m=model)
    n = size(y, 1)
    hist = zeros(n + 1)
    ŷ = m(x) .>= 0.5
    Δ = sum(abs.(ŷ - y), dims=1)
    for i = 1:n+1
        hist[i] = sum(Δ .== i-1)
    end
    return hist
end

hist = hist_dist(gpu(x_train_norm), gpu(y_train); m=model)
scatter(Vector(0:10), log.(hist .+ 1e-0))


loss(gpu(x_train_norm), gpu(y_train))
easy_loss(gpu(x_train_norm), gpu(y_train))
hist_dist(gpu(x_train_norm), gpu(y_train))

opt = Adam(0.001, (0.9, 0.999), 1.0e-8) # Gradient descent with learning rate 0.1
for epoch in 1:30
    for (x_train_batch, y_train_batch) in CuIterator(train_loader)
        Flux.train!(loss, parameters, [(x_train_batch, y_train_batch)], opt)
    end

    if (epoch - 1) % 10 == 0
        println(
            epoch, "   ",
            # round(loss(gpu(x_train_norm), gpu(y_train)), digits=4), "   ",
            round(loss(gpu(x_val_norm), gpu(y_val)), digits=4), "   ",
        )
    end
end
CUDA.memory_status()


loss(gpu(x_train_norm), gpu(y_train))
loss(gpu(x_val_norm), gpu(y_val))
loss(gpu(x_test_norm), gpu(y_test))
easy_loss(gpu(x_train_norm), gpu(y_train))
easy_loss(gpu(x_val_norm), gpu(y_val))
easy_loss(gpu(x_test_norm), gpu(y_test))

sum(hist_dist(gpu(x_train_norm), gpu(y_train), m=model)[1:1]) / H_train
sum(hist_dist(gpu(x_val_norm),   gpu(y_val),   m=model)[1:1]) / H_val
sum(hist_dist(gpu(x_test_norm),  gpu(y_test),  m=model)[1:1]) / H_test

sum(hist_dist(gpu(x_train_norm), gpu(y_train), m=model)[1:2]) / H_train
sum(hist_dist(gpu(x_val_norm),   gpu(y_val),   m=model)[1:2]) / H_val
sum(hist_dist(gpu(x_test_norm),  gpu(y_test),  m=model)[1:2]) / H_test

sum(hist_dist(gpu(x_train_norm), gpu(y_train), m=model)[1:3]) / H_train
sum(hist_dist(gpu(x_val_norm),   gpu(y_val),   m=model)[1:3]) / H_val
sum(hist_dist(gpu(x_test_norm),  gpu(y_test),  m=model)[1:3]) / H_test

hist_dist(x_train, y_train, m=easy_model)[1] / H_train
hist_dist(x_val, y_val, m=easy_model)[1] / H_train
hist_dist(x_test, y_test, m=easy_model)[1] / H_train

40*1 + 960 * 0.1

for i = 1:10
    @show i
    plt = plot()
    ŷ = Int.(round.(model(gpu(x_test)[:,i:i]), digits=0))
    scatter!(ŷ, color=:red)
    scatter!(y_test[:,i:i], color=:black)
    plot!(ŷ, color=:red)
    plot!(y_test[:,i:i], color=:black)
    display(plt)
    sleep(0.1)
end

function binary_projection(x, y, threshold; m=model)
    ŷ = m(x)
    ŷb = -1 * (ŷ .<= threshold) + 1 * (ŷ .>= 1-threshold)
    yb = -1 * (y .< 0.5) + 1 * (y .>= 0.5)
    opposite_error = abs.(ŷb - yb) .== 2
    indefinite_error = abs.(ŷb - yb) .== 1
    opposite_error = sum(opposite_error, dims=1)
    indefinite_error = sum(indefinite_error, dims=1)
    return mean(opposite_error .!= 0), mean(indefinite_error .!= 0)
end

binary_projection(gpu(x_train_norm), gpu(y_train), 0.05, m=model)
file_path = joinpath(@__DIR__, "model", "model0.bson")
BSON.@save file_path model
