using Plots
using Statistics
using Random

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# define mechanisms
################################################################################
A0 = [
    [
        +1.0 +0.2;
        -0.0 +1.0;
        -1.0 -0.3;
        +0.0 -1.0;
        +0.8 -0.8;
        ],
    ]
b0 = [
        0.40*[+1.0, +1.0, +1.0, +1.0, +1.0],
    ]

timestep = 0.02
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.20

mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )

bilevel_mech = get_bilevel_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=false,
        # complementarity_backstep=1e-1,
        )
    )



################################################################################
# simulation
################################################################################
H = 40

xp2  = [+0.00, +0.90, -0.00]
vp15 = [-0.00, +0.00, -3.00]
z0 = [xp2; vp15]

set_gravity!(mech, gravity)
set_gravity!(bilevel_mech, gravity)

Mehrotra.initialize_solver!(mech.solver)
Mehrotra.initialize_solver!(bilevel_mech.solver)

@elapsed storage = simulate!(mech, deepcopy(z0), H)
@elapsed bilevel_storage = simulate!(bilevel_mech, deepcopy(z0), H)

vis, anim = visualize!(vis, mech, storage, name=:single, color=RGBA(1,1,1,0.3))
vis, anim = visualize!(vis, bilevel_mech, bilevel_storage, animation=anim, name=:bilevel)

scatter(storage.iterations, color=:red)
scatter!(bilevel_storage.iterations, color=:blue)

################################################################################
# performance evaluation
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

function monte_carlo_performance_evaluation(mechanism::Mechanism{T},
        timesteps,
        complementarity_tolerances,
        initial_conditions,
        horizon;
        vis=Visualizer(),
        verbose=false) where T

    nt = length(timesteps)
    nc = length(complementarity_tolerances)
    ni = length(initial_conditions)
    performances = []
    N = length(initial_conditions)
    for i = 1:nt
        verbose && (@show i, nt)
        pj = []
        for j = 1:nc
            verbose && (@show j, nc)
            pk = []
            for k = 1:ni
                verbose && (@show i,j,k, nt,nc,ni)
                H = Int(floor(horizon / timesteps[i]))
                p = performance_evaluation(mechanism,
                    timesteps[i],
                    complementarity_tolerances[j],
                    initial_conditions[k],
                    H, vis=vis)
                push!(pk, p)
            end
            push!(pj, pk)
        end
        push!(performances, pj)
    end
    return performances
end


timesteps = 1 ./ [10, 20, 30, 50, 70, 100]
complementarity_tolerances = [1e-3, 3e-4, 1e-4, 1e-5, 1e-7, 1e-10]
z_min = [0.0, +sqrt(2)/2, 0, -1, -2, -1]
z_max = [0.0, +1.5, 2π, +1, +0, +1]
initial_conditions = generate_initial_conditions(20, z_min, z_max)


performance_evaluation(mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)
performance_evaluation(bilevel_mech, timesteps[1], complementarity_tolerances[1],
    initial_conditions[1], H, vis=vis)

horizon = 1.5
performances = monte_carlo_performance_evaluation(mech, timesteps,
    complementarity_tolerances, initial_conditions, horizon, vis=vis, verbose=true)
bilevel_performances = monte_carlo_performance_evaluation(bilevel_mech, timesteps,
    complementarity_tolerances, initial_conditions, horizon, vis=vis, verbose=true)


function save_pgf_matrix(A, offset, file_path)
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

folder_path = joinpath(@__DIR__, "data")

plt = process_performances(timesteps, complementarity_tolerances,
    initial_conditions, performances,
    offset=[0,0],
    suffix="single_level",
    folder_path=folder_path)
plt = process_performances(timesteps, complementarity_tolerances,
    initial_conditions, bilevel_performances,
    offset=[1+length(timesteps),0],
    suffix="bilevel",
    folder_path=folder_path)

# Fix bilevel contact with floor
# do no gravity polytope to polytope collision



A = rand(6,6)
offset = (6, 0)
# save_pgf_matrix(A, offset, file_path)
