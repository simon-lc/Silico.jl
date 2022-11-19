################################################################################
# utils
################################################################################
function generate_initial_conditions(N::Int, z_max::Vector{T}, z_min::Vector{T}; seed=0) where T
    Random.seed!(0)
    nz = length(z_min)
    initial_conditions = Vector{Vector{T}}()
    for i = 1:N
        z = z_min + rand(nz) .* (z_max .- z_min)
        push!(initial_conditions, z)
    end
    return initial_conditions
end

function save_pgf_matrix(A, offset, file_path)
    @show A
    n, m = size(A)
    io = open(file_path, "w")
    for i = 1:n
        for j = 1:m
            write(io, "$(offset[1] + i - 1 - 0.5)  $(offset[2] + j - 1 - 0.5)  $(A[i,j])\n")
            write(io, "$(offset[1] + i - 1 - 0.5)  $(offset[2] + j - 1 + 0.5)  $(A[i,j])\n")
            write(io, "$(offset[1] + i - 1 + 0.5)  $(offset[2] + j - 1 + 0.5)  $(A[i,j])\n")
            write(io, "$(offset[1] + i - 1 + 0.5)  $(offset[2] + j - 1 - 0.5)  $(A[i,j])\n")
        end
    end
    close(io)
    return nothing
end

function benchmark_evaluation(mechanism::Mechanism,
        timestep, complementarity_tolerance, initial_conditions,
        H; vis=Visualizer())

    set_timestep!(mechanism, timestep)
    mechanism.solver.options.complementarity_tolerance = complementarity_tolerance
    mechanism.solver.options.residual_tolerance = complementarity_tolerance / 10

    Mehrotra.initialize_solver!(mechanism.solver)
    try
        storage = simulate!(mechanism, copy(initial_conditions), H)
        visualize!(vis, mechanism, storage)
        return storage
    catch e
        @warn "Solver errored"
        errored = true
    end
    return nothing
end

function grid_benchmark_evaluation(mechanism::Mechanism{T},
        timesteps,
        complementarity_tolerances,
        initial_conditions,
        horizon;
        evaluation_metric=performance_evaluation,
        vis=Visualizer(),
        verbose=false) where T

    nt = length(timesteps)
    nc = length(complementarity_tolerances)
    ni = length(initial_conditions)
    results = []
    N = length(initial_conditions)
    for i = 1:nt
        verbose && (@show i, nt)
        rj = []
        for j = 1:nc
            verbose && (@show j, nc)
            rk = []
            for k = 1:ni
                verbose && (@show i,j,k, nt,nc,ni)
                H = Int(floor(horizon / timesteps[i]))
                r = evaluation_metric(mechanism,
                    timesteps[i],
                    complementarity_tolerances[j],
                    initial_conditions[k],
                    H, vis=vis)
                push!(rk, r)
            end
            push!(rj, rk)
        end
        push!(results, rj)
    end
    return results
end


################################################################################
# momentum
################################################################################
function momentum(mechanism::Mechanism, storage::TraceStorage{T,H}) where {T,H}
    m = zeros(3,H)

    for i = 1:H
        for (j,body) in enumerate(mechanism.bodies)
            v = storage.v[i][j]
            M = Diagonal([body.mass[1], body.mass[1], body.inertia[1]])
            m[:,i] += M * v
        end
    end
    return m
end

function momentum_evaluation(mechanism::Mechanism,
        timestep, complementarity_tolerance, initial_conditions,
        H; vis=Visualizer())

    storage = benchmark_evaluation(mechanism,
            timestep, complementarity_tolerance, initial_conditions,
            H; vis=vis)

    errored = false
    m = 0.0
    if storage != nothing
        m = momentum(mechanism, storage)
        linear_momentum = mean(abs.(m[1:2,:]))
        angular_momentum = mean(abs.(m[3,:]))
    else
        @warn "Solver errored"
        errored = true
    end
    return linear_momentum, angular_momentum, errored
end

function process_momentum(timesteps, complementarity_tolerances, initial_conditions,
        results; offset=[0,0],
        suffix="",
        folder_path=@__DIR__)

    nt = length(timesteps)
    nc = length(complementarity_tolerances)
    ni = length(initial_conditions)

    linear_momentum = zeros(nt, nc)
    angular_momentum = zeros(nt, nc)
    error_rate = zeros(nt, nc)
    for i = 1:nt
        for j = 1:nc
            lin = [results[i][j][k][1] for k=1:ni]
            ang = [results[i][j][k][2] for k=1:ni]
            errored = [results[i][j][k][3] for k=1:ni]

            linear_momentum[i,j] = sum(lin) / ni
            angular_momentum[i,j] = sum(ang) / ni
            error_rate[i,j] = sum(errored) / ni
        end
    end
    @show linear_momentum
    @show size(linear_momentum)
    save_pgf_matrix(linear_momentum, offset, joinpath(folder_path, "linear_momentum_$suffix.dat"))
    save_pgf_matrix(angular_momentum, offset, joinpath(folder_path, "angular_momentum_$suffix.dat"))
    save_pgf_matrix(error_rate, offset, joinpath(folder_path, "error_rate_$suffix.dat"))

    @show mean(linear_momentum)
    plt = heatmap(linear_momentum, title="linear_momentum")
    display(plt)
    plt = heatmap(angular_momentum, title="angular_momentum")
    display(plt)
    plt = heatmap(error_rate, title="error_rate")
    display(plt)
    return plt
end

################################################################################
# performance
################################################################################
mutable struct Performance1130{T}
    violation::T
    iterations::T
    solve_failed::Bool
    solve_errored::Bool
end

function performance_evaluation(mechanism::Mechanism,
        timestep, complementarity_tolerance, initial_conditions,
        H; vis=Visualizer())

    storage = benchmark_evaluation(mechanism,
            timestep, complementarity_tolerance, initial_conditions,
            H; vis=vis)

    max_iterations = mechanism.solver.options.max_iterations
    if storage != nothing
        errored = false
        # iterations
        iterations = mean(storage.iterations)
        # failure
        failed = any(storage.iterations .== max_iterations)
        # violation
        violation = contact_violation(mechanism, storage)
        return Performance1130(violation, iterations, failed, errored)
    else
        @warn "Solver errored"
        errored = true
        # iterations
        iterations = max_iterations
        # failure
        failed = true
        # violation
        violation = 0.0
        return Performance1130(violation, iterations, failed, errored)
    end
end

function contact_violation(mechanism::Mechanism, storage::TraceStorage{T,H}) where {T,H}
    violation = 0.0

    for contact in mechanism.contacts
        parent_shape = contact.parent_shape
        child_shape = contact.child_shape
        detector = CollisionDetector(parent_shape, child_shape)
        if (typeof(contact)<:EnvContact2D) || (typeof(contact)<:EnvBilevelContact2D)
            idx = find_body_index(mechanism.bodies, contact.parent_name)
            for t = 1:H
                ϕ = contact_data(storage.x[t][idx], zeros(3), detector)[1]
                violation = max(violation, max(-ϕ[1], 0.0))
            end
        end
    end
    return violation
end

function process_performances(timesteps, complementarity_tolerances, initial_conditions,
        performances; offset=[0,0],
        suffix="",
        folder_path=@__DIR__)

    nt = length(timesteps)
    nc = length(complementarity_tolerances)
    ni = length(initial_conditions)

    violation_rate = zeros(nt, nc)
    iterations = zeros(nt, nc)
    failure_rate = zeros(nt, nc)
    error_rate = zeros(nt, nc)
    for i = 1:nt
        for j = 1:nc
            violation = getfield.(performances[i][j], :violation)
            iter = getfield.(performances[i][j], :iterations)
            failed = getfield.(performances[i][j], :solve_failed)
            errored = getfield.(performances[i][j], :solve_errored)
            violation_rate[i,j] = sum(violation) / ni
            iterations[i,j] = sum(iter) / ni
            failure_rate[i,j] = sum(failed) / ni
            error_rate[i,j] = sum(errored) / ni
        end
    end

    save_pgf_matrix(failure_rate, offset, joinpath(folder_path, "failure_rate_$suffix.dat"))
    save_pgf_matrix(violation_rate, offset, joinpath(folder_path, "violation_rate_$suffix.dat"))
    save_pgf_matrix(iterations, offset, joinpath(folder_path, "iterations_$suffix.dat"))
    save_pgf_matrix(error_rate, offset, joinpath(folder_path, "error_rate_$suffix.dat"))

    plt = heatmap(violation_rate, title="violation rate")
    display(plt)
    plt = heatmap(iterations, title="iterations")
    display(plt)
    plt = heatmap(failure_rate, title="failure rate")
    display(plt)
    plt = heatmap(error_rate, title="error rate")
    display(plt)
    return plt
end
