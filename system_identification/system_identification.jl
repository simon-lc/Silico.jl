using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using Clustering
using ForwardDiff
using BenchmarkTools

vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

include("../src/DojoLight.jl")

include("../cvxnet/softmax.jl")
include("../cvxnet/point_cloud.jl")
include("../cvxnet/halfspace.jl")
include("../cvxnet/visuals.jl")

include("../system_identification/newton_solver.jl")
include("../system_identification/structure.jl")
include("../system_identification/methods.jl")
include("../system_identification/visuals.jl")


################################################################################
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
################################################################################
# partial observability: we currently observe the whole state
# works with multiple bodies and especially finger that are of known geometry
# need to integrate sphere to the point cloud generator
# speedup
# gradient and hessian approximated
# generic optimizer
# work with arbitrary number of shapes and bodies


################################################################################
# demo
################################################################################
timestep = 0.10;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.05

mech = get_polytope_drop(;
# mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    options=Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );

################################################################################
# test simulation
################################################################################
xp2 = [-0.80, +0.5, +0.0]
vp15 = [+1,0, -0.0]
z0 = [xp2; vp15]
H0 = 20
storage = simulate!(mech, z0, H0)
zf = get_current_state(mech)

eye_position = [-0.0, 3.0]
num_points = 50
camera_rays = range(-0.3π, -0.7π, length=num_points)
look_at = [0.0, 0.0]
softness = 2.5
cameras = [Camera1320(eye_position, camera_rays, look_at, softness)]
context = CvxContext1320(mech, cameras)
state = z0

measurements = simulate!(context, state, H0)
vis, anim = visualize!(vis, context, measurements)


################################################################################
# test filtering objective
################################################################################
P_state = Diagonal(ones(6))
M_observation = Diagonal(ones(6))
P_parameters = Diagonal([1e2*ones(3); 1e-4*ones(14)])
M_point_cloud = 1.0
obj = CvxObjective1320(P_state, M_observation, P_parameters, M_point_cloud)


state_prior = deepcopy(zf)
state_guess = deepcopy(zf)
params_prior = get_parameters(context)
params_guess = get_parameters(context)

c, g, H = filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, deepcopy(measurements[end]))

# @benchmark filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, deepcopy(measurements[end]))

# Main.@profiler [filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, deepcopy(measurements[end])) for i=1:100]




################################################################################
# one step filtering demo
################################################################################
# context
context = deepcopy(CvxContext1320(mech, cameras))

# objective
P_state = Diagonal(1e+2ones(6))
M_observation = 1e-2Diagonal(ones(6))
P_parameters = Diagonal([1e2*ones(3); 1e-4*ones(14)])
M_point_cloud = 1e-0
obj = CvxObjective1320(P_state, M_observation, P_parameters, M_point_cloud)

# prior
state_prior = deepcopy(zf + 3e-2 * (rand(6) .- 0.5))
params_prior = get_parameters(context)
params_prior[4:end] .+= 5e-1 * (rand(14) .- 0.5)

# guess
state_guess = deepcopy(zf)
params_guess = get_parameters(context)
# params_guess.θ[4:end] .+= 0.01

# truth
state_truth = deepcopy(zf)
params_truth = get_parameters(context)

# measurement
measurement = measurement_model(context, state_truth, params_truth)
measurement.z .+=  5e-2(rand(6) .- 0.5)


filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurement)

function local_filtering_objective(x)
    nz = context.mechanism.dimensions.state
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz+1:end])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return c
end

function local_filtering_gradient(x)
    nz = context.mechanism.dimensions.state
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz+1:end])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return g
end

function local_filtering_hessian(x)
    nz = context.mechanism.dimensions.state
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz+1:end])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return H
end

x0 = [state_prior; params_prior]
local_filtering_objective(x0)
local_filtering_gradient(x0)
local_filtering_hessian(x0)
local_projection100(x) = projection(context, x)
local_clamping = x -> x

xsol, xtrace = newton_solver!(x0,
    local_filtering_objective,
    local_filtering_gradient,
    local_filtering_hessian,
    local_projection100,
    local_clamping,
    residual_tolerance=1e-4,
    reg_min=1e-0,
    reg_max=1e+3,
    reg_step=1.3,
    max_iterations=20,
    line_search_iterations=10,
    )

# plot(hcat(xtrace...)')

# vis = Visualizer()
# render(vis)
# set_floor!(vis, color=RGBA(0,0,0,0.4))
# set_light!(vis)
# set_background!(vis)

prior = [state_prior; params_prior]
solution = [state_truth; params_truth]
vis, anim = visualize_solve!(vis, context, prior, solution, xtrace)




################################################################################
# multi step filtering demo
################################################################################

# vis = Visualizer()
# render(vis)
# set_floor!(vis, color=RGBA(0,0,0,0.4))
# set_light!(vis)
# set_background!(vis)

# generate reference trajectory

# context
context = deepcopy(CvxContext1320(mech, cameras))

# objective
P_state = Diagonal(1e+2ones(6))
M_observation = 1e+2Diagonal(ones(6))
P_parameters = Diagonal([1e2*ones(3); 1e-2*ones(14)])
M_point_cloud = 1e-0
obj = CvxObjective1320(P_state, M_observation, P_parameters, M_point_cloud)

# local functions
local_projection100(x) = projection(context, x)
local_clamping = x -> x

# truth
params_truth = get_parameters(context)

# prior
state_prior = deepcopy(measurements[1].z)
params_truth = get_parameters(context)
params_prior = deepcopy.(params_truth)
params_prior[4:end] .+= 2e-1 * ([1,0,1,1,1,0,0,1,0,1,1,1,0,1] .- 0.5)

x_prior = deepcopy([state_prior; params_prior])
x = deepcopy([measurements[2].z; params_prior])

for i = 2:20
    println("step $i")
    nz = context.mechanism.dimensions.state

    # cheating
    x_prior[1:nz] .= deepcopy(measurements[i-1].z)
    # x_prior[nz+1:end] .= deepcopy(params_truth)
    x .= deepcopy(x_prior)
    x[1:nz] .= deepcopy(measurements[i].z)

    # x[nz+1:end] .= deepcopy(params_truth)


    function local_filtering_objective(x)
        nz = context.mechanism.dimensions.state
        state_guess = deepcopy(x[1:nz])
        params_guess = deepcopy(x[nz+1:end])
        state_prior = deepcopy(x_prior[1:nz])
        params_prior = deepcopy(x_prior[nz+1:end])

        c, g, H = filtering_objective(context, obj, state_guess, params_guess,
            state_prior, params_prior, measurements[i])
        return c, g, H
    end

    local_objective(x) = local_filtering_objective(x)[1]
    local_gradient(x) = local_filtering_objective(x)[2]
    local_hessian(x) = local_filtering_objective(x)[3]

    x_sol, x_trace = newton_solver!(x,
        local_objective,
        local_gradient,
        local_hessian,
        local_projection100,
        local_clamping,
        residual_tolerance=1e-3,
        reg_min=1e-0,
        reg_max=1e+3,
        reg_step=1.3,
        max_iterations=5,
        line_search_iterations=10,
        )

    vis, anim = visualize_solve!(vis, context, x_prior,
        [deepcopy(measurements[i].z); params_truth], x_trace)
    sleep(2.0)
    x_prior = deepcopy(x_sol)
end


predicted_measurement = measurement_model(context, measurements[2].z, params_truth)
measurements[2].z - predicted_measurement.z
measurements[2].d - predicted_measurement.d




# vis = Visualizer()
# render(vis)
# set_floor!(vis, color=RGBA(0,0,0,0.4))
# set_light!(vis)
# set_background!(vis)
