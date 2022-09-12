################################################################################
# structure
################################################################################
# measurement obtained at each time step
abstract type Measurement{T} end

# time-varying state of the system
abstract type State{T} end

# parameters of the system that are fixed in time and that we want to estimate
abstract type Parameters{T} end

# all ground-truth information including time-invariant parameters that we want to estimate (parameters)
# and that we assume are known (camera position for instance).
# it also contains the state of the system
abstract type Context{T} end

# objective that we want to minimize to regress correct states and parameters
abstract type Objective{T} end

# find the best fitting states and parameters given a sequence of measurement and a prior over the state and parameters
# it contains the context and the objective
abstract type Optimizer{T} end




################################################################################
# Cvx instantiation of System Identification
################################################################################

## measurement
struct CvxMeasurement1320{T} <: Measurement{T}
    z::Vector{T} # polytope state, we assume the whole state is observed TODO
    d::Vector{Matrix{T}} # point cloud obtained from several cameras
end

function pack(mass, inertia, friction, A, b, o)
    θ_inertial = [mass; inertia; friction_coefficient]
    θ_geometry, _ = pack_halfspaces(A, b, o)
    return [θ_inertial; θ_geometry]
end

function unpack(θ::Vector, polytope_dimensions::Vector{Int})
    off = 0
    mass, inertia, friction_coefficient = unpack_dynamics(θ[1:3]); off += 3
    A, b, o = unpack_halfspaces(θ[off + 1:end], polytope_dimensions)
    return mass, inertia, friction_coefficient, A, b, o
end

function unpack_inertial(θ)
    mass = θ[1]
    inertia = θ[2]
    friction_coefficient = θ[3]
    return mass, inertia, friction_coefficient
end

# camera
struct Camera1320{T}
    eye_position::Vector{T}
    camera_rays
    look_at::Vector{T}
    softness::T
end

## context
struct CvxContext1320{T} <: Context{T}
    mechanism::Mechanism
    cameras::Vector{Camera1320{T}}
    polytope_dimensions::Vector{Int}
end

function CvxContext1320(mechanism::Mechanism, cameras::Vector{Camera1320{T}}) where T
    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ_geometry, polytope_dimensions = pack_halfspaces(A, b, o)
    return CvxContext1320(mechanism, cameras, polytope_dimensions)
end

function unpack(θ, context::CvxContext1320)
    polytope_dimensions = context.polytope_dimensions
    return unpack(θ, polytope_dimensions)
end

## objective
struct CvxObjective1320{T} <: Objective{T}
    P_state::Diagonal{T, Vector{T}} # state prior regularizing cost
    M_observation::Diagonal{T, Vector{T}} # observation measurement regularizing cost
    P_parameters::Diagonal{T, Vector{T}} # parameters prior regularizing cost
    M_point_cloud::T # point cloud measurement regularizing cost
end

## optimizer
struct CvxOptimizer1320{T} <: Optimizer{T}
    context::CvxContext1320{T}
    objective::CvxObjective1320{T}
end
