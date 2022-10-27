using Flux
using Flux: update!

W = rand(2, 5)
b = rand(2)



predict(x) = (W * x) .+ b
function loss(x, y)
    ŷ = predict(x)
    Flux.Losses.mse(ŷ, y, agg=mean)
end

D = 100
x, y = rand(5,D), rand(2,D) # Dummy data
l = loss(x, y) # ~ 3



θ = Flux.params(W, b)
grads = gradient(() -> loss(x, y), θ)

opt = Adam(0.001, (0.9, 0.999), 1.0e-8) # Gradient descent with learning rate 0.1
data = [(x, y)]
for i = 1:1000
    Flux.train!(loss, θ, data, opt)
end
l = loss(x, y)



D = 10000
x, y = rand(5,D), rand(2,D) # Dummy data
data = [(x, y)]

model = Chain(Dense(5 => 4, relu), Dense(4 => 2));
parameters = Flux.params(model)
loss(x, y) = Flux.Losses.mse(model(x), y, agg=mean)
opt = Adam(0.001, (0.9, 0.999), 1.0e-8) # Gradient descent with learning rate 0.1



loss(x, y)
for epoch in 1:12000
    Flux.train!(loss, parameters, data, opt)
end
loss(x, y)
