################################################################################
# bilevel formulation
################################################################################
function contact_data(parent_pose, child_pose, detector::CollisionDetector{T,D,NC,NP}) where {T,D,NC,NP}
    detection = detector.detection
    solver = detector.solver
    solver.options.differentiate = true

    detection.parent_pose .= parent_pose
    detection.child_pose .= child_pose
    θ = get_parameters(detection)
    solver.parameters .= θ

    Mehrotra.solve!(solver)
    c, α, βp, βc, λα, λp, λc, sα, sp, sc = unpack_variables(solver.solution.all, detection)

    ϕ = α .- 1
    contact_w = c + (parent_pose + child_pose)[1:2] / 2

    dϕdp = solver.data.solution_sensitivity[3:3,1:3]
    dcdp = solver.data.solution_sensitivity[1:2,1:3]

    return ϕ, contact_w, dϕdp, dcdp
end

function bilevel_contact_data(variables, parameters, contact)
    # variables
    primals = variables[1:3]

    # body dynamics
    v25 = primals
    # parameters
    timestep, p2, v15 = parameters[1], parameters[2:4], parameters[5:7]
    # integrator
    p3 = p2 + timestep * v25

    ϕ, contact_point, dϕdp3, dcdp3 = contact_data(p3, zeros(3), contact.detector)
    return ϕ, contact_point, dϕdp3, dcdp3
end

function bilevel_residual(variables, parameters, ϕ, contact_point, contact)
    # problem data
    gravity = -9.81
    mass = 1.0
    inertia = 0.2 * ones(1,1)
    friction_coefficient = 0.1

    # variables
    primals = variables[1:3]
    duals = variables[4:7]
    slacks = variables[8:11]

    # body dynamics
    v25 = primals
    # parameters
    timestep, p2, v15 = parameters[1], parameters[2:4], parameters[5:7]
    # integrator
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])

    # body dynamics
    body_optimality = M * (p3 - 2*p2 + p1)/timestep - timestep * [0; mass .* gravity; 0]

    # contact dynamics
    # unpack parameters
    γ, ψ, β = duals[1:1], duals[2:2], duals[3:4]
    sγ, sψ, sβ = slacks[1:1], slacks[2:2], slacks[3:4]

    # ϕ, contact_point, dϕdp3, dcdp3 = contact_data(p3, zeros(3), contact.detector)
    normal_pw = [0.0, 1.0]
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_point - p3[1:2]; 0]) * [f_pw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]

    # tangential velocities at the contact point
    tanvel_p = v25[1:2] + (skew([p3[1:2] - contact_point; 0]) * [zeros(2); v25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel = tanvel_p

    # @show ϕ, contact_point
    slackness = [
        sγ - ϕ;
        sψ - (friction_coefficient * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
    ]

    res = [body_optimality - wrench_p; slackness]
    return res
end

function bilevel_residual(variables, parameters, contact)
    ϕ, contact_point, dϕdp3, dcdp3 = bilevel_contact_data(variables, parameters, contact)
    return bilevel_residual(variables, parameters, ϕ, contact_point, contact)
end

function bilevel_residual_jacobian(variables, parameters, contact)
    timestep = parameters[1]
    num_variables = length(variables)
    dp3dx = zeros(3, num_variables)
    dp3dx[1:3,1:3] += timestep * I(3)

    ϕ, contact_point, dϕdp3, dcdp3 = bilevel_contact_data(variables, parameters, contact)
    dϕdx = dϕdp3 * dp3dx
    dcdx = dcdp3 * dp3dx

    drdx = ForwardDiff.jacobian(
        variables -> bilevel_residual(variables, parameters, ϕ, contact_point, contact),
        variables)

    drdϕ = ForwardDiff.jacobian(
        ϕ -> bilevel_residual(variables, parameters, ϕ, contact_point, contact),
        ϕ)

    drdc = ForwardDiff.jacobian(
        contact_point -> bilevel_residual(variables, parameters, ϕ, contact_point, contact),
        contact_point)

    DrDx = drdx + drdϕ * dϕdx + drdc * dcdx
    return DrDx
end

function bilevel_methods(contact::EnvBilevelContact2D, dim::Mehrotra.Dimensions, idx::Mehrotra.Indices)
    parameter_keywords = idx.parameter_keywords

    # in-place evaluation
    function equality_constraint(out, x, θ)
        out .= bilevel_residual(x, θ, contact)
    end

    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        @warn warning
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        matrix_cache .= bilevel_residual_jacobian(x, θ, contact)

        matrix_cache[idx.primals, idx.primals] .+= 1e-6*Diagonal(ones(dim.primals))
        matrix_cache[idx.duals, idx.duals] .-= 1e-8*Diagonal(ones(dim.duals))
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end

    # correction
    function correction(c, r, Δza, Δsa, κ)
        c .= r
        c[idx.cone_product] .-= (κ - Mehrotra.cone_product(Δza, Δsa, idx.cone_nonnegative, idx.cone_second_order))
        return nothing
    end

    # correction compressed
    function correction_compressed(cc, r, Δza, Δsa, x, κ)
        error(warning)
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        error(warning)
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        error(warning)
    end

    equality_jacobian_keywords = Vector{Function}()
    for k in eachindex(parameter_keywords)
        function func(vector_cache, x, θ)
            error(warning)
        end
        push!(equality_jacobian_keywords, func)
    end

    ex_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(Mehrotra.SparseArrays.sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(Mehrotra.SparseArrays.sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([Mehrotra.SparseArrays.findnz(Mehrotra.SparseArrays.sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))
    eθ_indices = zeros(Int, dim.equality, dim.parameters)
    vec(eθ_indices) .= 1:length(eθ_indices)
    ek_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        indices = vec(eθ_indices[:,val])
        push!(ek_indices, indices)
    end

    methods = Mehrotra.ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        equality_jacobian_keywords,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_indices,
    )

    return methods
end

function get_next_state(solver)
    v25 = solver.solution.primals
    timestep = solver.parameters[1]
    p2 = solver.parameters[2:4]
    p3 = p2 + timestep * v25
    return [p3; v25]
end

function bilevel_simulate!(solver, z0, H::Int)
    solve_times = []
    Z = []
    z = deepcopy(z0)
    for i = 1:H
        push!(Z, z)
        p2, v15 = z[1:3], z[4:6]
        solver.parameters[2:4] .= p2
        solver.parameters[5:7] .= v15
        t = @elapsed Mehrotra.solve!(solver)
        push!(solve_times, t)
        z = get_next_state(solver)
    end
    return Z, solve_times
end

function visualize!(vis::Visualizer, Z, A, b, timestep;
        name=:robot,
        animation=MeshCat.Animation(Int(floor(1/timestep))))

    build_2d_polytope!(vis[name][:body], A, b)
    for i = 1:length(Z)
        z = Z[i]
        atframe(animation, i) do
            set_2d_polytope!(vis[name], z[1:2], z[3:3], name=:body)
        end
    end
    setanimation!(vis, animation)
    return vis, animation
end

function initial_conditions(z_min, z_max, N::Int, seed=0)
    Random.seed!(seed)
    n = length(z_min)
    Z_init = []
    for i = 1:N
        z = z_min + rand(n) .* (z_max - z_min)
        push!(Z_init, z)
    end
    return Z_init
end

function benchmark_bilevel(solver, Z_init, H, visualize!)
    solve_times = []
    for z in Z_init
        Z_traj, s = bilevel_simulate!(solver, z, H)
        visualize!(vis, Z_traj)
        push!(solve_times, s)
    end
    return solve_times
end


################################################################################
# single-level formulation
################################################################################
function step!(mechanism::Mechanism, z0)
    set_current_state!(mechanism, z0)
    nu = sum(input_dimension.(mech.bodies))
    set_input!(mechanism, zeros(nu))
    update_parameters!(mechanism)
    t = @elapsed Mehrotra.solve!(mechanism.solver)
    update_nodes!(mechanism)
    z1 = get_next_state(mechanism)
    return z1, t
end

function simulate!(mechanism::Mechanism{T}, z0, H::Int;
        controller::Function=(m,i)->nothing) where T

    solve_times = []
    storage = TraceStorage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z, t = step!(mechanism, z)
        record!(storage, mechanism, i)
        push!(solve_times, t)
    end
    return storage, solve_times
end

function benchmark_single_level(mechanism::Mechanism, Z_init, H)
    solve_times = []
    for z in Z_init
        storage, s = simulate!(mechanism, z, H)
        visualize!(vis, mechanism, storage)
        push!(solve_times, s)
    end
    return solve_times
end

single_level_solve_times = benchmark_single_level(mech, Z_init, H)
single_level_solve_times = vcat(single_level_solve_times...)
histogram(single_level_solve_times)

using ForwardDiff
using Random
using Plots

################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

################################################################################
# problem data
################################################################################
timestep = 0.01
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
friction_coefficient = 0.1
A = [1 0; -1 0; 0 1; 0 -1.0]
b = 0.5 * [1,1,1,1.0]

################################################################################
# contact
################################################################################
shapes = [PolytopeShape(A, b)]
body = Body(timestep, mass, inertia, shapes, gravity=gravity, name=:body)
shape = HalfspaceShape([0, 1.0])
contact = EnvBilevelContact2D(body, shape;
        name=:contact,
        complementarity_tolerance=1e-5,
        friction_coefficient=friction_coefficient)

################################################################################
# parameters
################################################################################
p2 = [0,3,0.0]
v15 = [0,0,0.0]
parameters = [timestep; p2; v15]

################################################################################
# methods
################################################################################
num_primals = 3
num_cone = 4
num_parameters = 7
dim = Dimensions(num_primals, num_cone, num_parameters)
idx = Indices(num_primals, num_cone, num_parameters)
methods = bilevel_methods(contact, dim, idx)

solver = Mehrotra.Solver(nothing, num_primals, num_cone;
    parameters=parameters,
    methods=methods,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-5,
        residual_tolerance=1e-6,
        compressed_search_direction=false,
        sparse_solver=false,
        differentiate=false,
    ),
)

Mehrotra.solve!(solver)

################################################################################
# simulate
################################################################################
z0 = [0, 3, 0, 0, 0, 10.0]
H = 1000
Z, solve_times = bilevel_simulate!(solver, z0, H)
mean(solve_times)
plot(hcat(Z...)')
visualize!(vis, Z, A, b, timestep)

################################################################################
# benchmark solve times
################################################################################
z_min = [0.0, 1.0, -3.0, -2, -2, -2]
z_max = [0.0, 3.0, +3.0, +2, +2, +2]
N = 100
H = 200
Z_init = initial_conditions(z_min, z_max, N)
fct_vis(vis, Z_traj) = visualize!(vis, Z_traj, A, b, timestep)
bilevel_solve_times = benchmark_bilevel(solver, Z_init, H, fct_vis)


################################################################################
# single-level method
################################################################################
mech = get_polytope_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    A=A,
    b=b,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-5,
        compressed_search_direction=true,
        sparse_solver=false,
        warm_start=false,
        differentiate=false,
        )
    );

single_level_solve_times = benchmark_single_level(mech, Z_init, H)


################################################################################
# process data
################################################################################
plt = plot(layout=(1,2))

bilevel_solve_times = vcat(bilevel_solve_times...)
histogram!(plt, bilevel_solve_times, color=:blue, xlims=[0,0.025], subplot=1)

single_level_solve_times = vcat(single_level_solve_times...)
histogram!(plt, single_level_solve_times, color=:red, subplot=2)

function process_solve_times(solve_times; s_max=0.01, n_buckets=20)
    solve_times = vcat(solve_times...)
    y_axis = zeros(n_buckets+1)
    x_axis = [i * s_max / n_buckets for i=0:n_buckets]
    for i = 1:n_buckets
        s_low = (i-1) * s_max / n_buckets
        s_high = i * s_max / n_buckets
        y_axis[i] = sum((solve_times .>= s_low) .* (solve_times .< s_high))
    end
    for i = 1:n_buckets+1
        println("($(1000*x_axis[i]), $(y_axis[i]))")
    end
    return x_axis, y_axis
end

process_solve_times(bilevel_solve_times, s_max=0.01, n_buckets=20)
process_solve_times(single_level_solve_times, s_max=0.0005, n_buckets=20)
mean(bilevel_solve_times) / mean(single_level_solve_times)
