################################################################################
# dataset
################################################################################

function extract_feature_label(mechanism, storage::TraceStorage{T,H}) where {T,H}
    solver = mechanism.solver
    data = solver.data
    problem = solver.problem
    indices = solver.indices
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution

    idx_duals = indices.duals
    idx_slacks = indices.slacks

    y = []
    x = []
    for i = 2:H
        previous_variables = storage.variables[i-1]
        variables = storage.variables[i]
        previous_parameters = storage.parameters[i-1]
        parameters = storage.parameters[i]
        solution.all .= previous_variables

        Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
            equality_constraint=true,
            equality_jacobian_variables=false,
            equality_jacobian_parameters=true,
            cone_constraint=false,
            cone_jacobian=false,
            sparse_solver=false,
            compressed=true)
        Mehrotra.residual!(data, problem, indices,
            residual=true,
            jacobian_variables=true,
            compressed=true,
            sparse_solver=false)

        previous_residual = deepcopy(data.residual.all)
        # previous_jacobian = vec(deepcopy(data.jacobian_variables_compressed_dense))
        previous_active_set = previous_variables[idx_duals] .>= previous_variables[idx_slacks]
        active_set = variables[idx_duals] .>= variables[idx_slacks]
        previous_log_variables = log.(10, previous_variables[[idx_duals; idx_slacks]])
        δ_parameters = parameters - previous_parameters
        # dd = data.jacobian_parameters * δ_parameters

        yi = [previous_active_set;
            # previous_log_variables;
            # previous_jacobian;
            # dd;
            previous_variables;
            previous_residual;
            δ_parameters]
        xi = active_set
        push!(y, yi)
        push!(x, xi)
    end
    y = hcat(y...)
    x = hcat(x...)
    return y, x
end


function save_dataset(x_train, y_train, x_val, y_val, x_test, y_test, μ, σ; name="dataset")
    jldsave(joinpath(@__DIR__, "data", "$name.jld2");
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        μ=μ,
        σ=σ)
    return nothing
end

function load_dataset(; name="dataset")
    file = jldopen(joinpath(@__DIR__, "data", "$name.jld2"), "r")
    return file["x_train"], file["y_train"],
        file["x_val"], file["y_val"],
        file["x_test"], file["y_test"],
        file["μ"], file["σ"]
end


################################################################################
# model
################################################################################

function train_model!(train_loader, loss, parameters, optimizer, n_epoch;
        validation_loss=f()=0,
        print_epoch=5
        )

    for epoch in 1:n_epoch
        for (x_batch, y_batch) in CuIterator(train_loader)
            Flux.train!(loss, parameters, [(x_batch, y_batch)], optimizer)
        end

        if (epoch - 1) % print_epoch == 0
            println(
                "epoch: ", epoch,
                "     val: ", validation_loss(),
            )
        end
    end
    return nothing
end

function save_model(model; name="model")
    file_path = joinpath(@__DIR__, "model", "$name.bson")
    BSON.@save file_path model = cpu(model)
    return nothing
end

function load_model(; name="model")
    file_path = joinpath(@__DIR__, "model", "$name.bson")
    BSON.@load file_path model
    return model
end


################################################################################
# performance analysis
################################################################################

function error_distribution(x, y; m=x->x)
    n = size(y, 1)
    dist = zeros(n + 1)
    ŷ = m(x) .>= 0.5
    Δ = sum(abs.(ŷ - y), dims=1)
    for i = 1:n+1
        dist[i] = sum(Δ .== i-1)
    end
    return dist
end

function binary_projection(x, y, threshold; m=x->x)
    ŷ = m(x)
    ŷb = -1 * (ŷ .<= threshold) + 1 * (ŷ .>= 1-threshold)
    yb = -1 * (y .< 0.5) + 1 * (y .>= 0.5)
    opposite_rate = abs.(ŷb - yb) .== 2
    indefinite_rate = abs.(ŷb - yb) .== 1
    opposite_rate = sum(opposite_rate, dims=1)
    indefinite_rate = sum(indefinite_rate, dims=1)
    return mean(opposite_rate .!= 0), mean(indefinite_rate .!= 0)
end
