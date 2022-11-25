################################################################################
# dataset
################################################################################

function data_collection_controller(mechanism, i)
    v = mechanism.solver.solution.primals[1:3]
    p = mechanism.solver.parameters[1:3]
    u_prev = mechanism.solver.parameters[7:9]
    u = 0.8 * u_prev .+ [15, 3, 15] .* (rand(3) .- [0.5, 0.5, 0.25]) - 1v - [1, 0, 0] .* p
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    return nothing
end

function extract_feature_label(mechanism, storage::TraceStorage{T,H}) where {T,H}
    x = []
    y = []
    for i = 2:H
        xi, yi = extract_feature_label(mechanism, storage, i)
        push!(x, xi)
        push!(y, yi)
    end
    x = hcat(x...)
    y = hcat(y...)
    return x, y
end

function extract_feature_label(mechanism, storage::TraceStorage{T,H}, i) where {T,H}
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
    # previous_log_variables = log.(10, previous_variables[[idx_duals; idx_slacks]])
    previous_active_set = previous_variables[idx_duals] .>= previous_variables[idx_slacks]
    active_set = variables[idx_duals] .>= variables[idx_slacks]
    δ_parameters = parameters - previous_parameters
    δ_residual = data.jacobian_parameters * δ_parameters
    # @show round.(δ_residual, digits=2)
    # @show data.jacobian_parameters
    # @show round.(δ_parameters, digits=4)
    # @show round.(δ_residual, digits=4)

    xi = [previous_active_set;
        # previous_log_variables;
        # previous_jacobian;
        previous_variables;
        unoptimized_residual[idx_equality]; # the idx_complementarity always = 0
        δ_residual[idx_equality]; # the idx_complementarity always = 0
        δ_parameters]
    yi = active_set
    return xi, yi
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
    model = BSON.load(file_path, @__MODULE__)[:model]
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


################################################################################
# masking
################################################################################


function residual(variables, parameters, solver)
    data = solver.data
    problem = solver.problem
    indices = solver.indices
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution

    idx_duals = indices.duals
    idx_slacks = indices.slacks


    solution.all .= variables
    solver.parameters .= parameters
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        sparse_solver=false,
        compressed=false)
    Mehrotra.residual!(data, problem, indices,
        residual=true,
        jacobian_variables=false,
        compressed=false,
        sparse_solver=false)

    return data.residual.all
end

function residual_jacobian(variables, parameters, solver)
    data = solver.data
    problem = solver.problem
    indices = solver.indices
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution

    idx_duals = indices.duals
    idx_slacks = indices.slacks


    solution.all .= variables
    solver.parameters .= parameters
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=false,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        sparse_solver=false,
        compressed=false)
    Mehrotra.residual!(data, problem, indices,
        residual=false,
        jacobian_variables=true,
        compressed=false,
        sparse_solver=false)

    return data.jacobian_variables_dense
end
