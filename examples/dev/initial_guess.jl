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
# define mechanism
################################################################################
timestep = 0.02;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.9;

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
        complementarity_tolerance=1e-3,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=true,
        warm_start=true,
        # complementarity_backstep=1e-1,
        )
    )

floor_contact = bilevel_mech.contacts[1]
floor_contact.detector.solver.options.complementarity_tolerance


mech = get_polytope_collision(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    A=A0, b=b0,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        complementarity_backstep=1e-2,
        # compressed_search_direction=true,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        )
    );

mech.solver.options.complementarity_backstep


function step!(mechanism::Mechanism, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)

    # get pose of the polytope
    parent_pose = get_next_pose(mechanism.solver.solution.all, mechanism.bodies[1])
    # solve the convex contact
    contact_initializer!(parent_pose, zeros(3), floor_contact.detector)
    # set the initialization
    idx_primals = mechanism.contacts[1].index.primals
    idx_duals = mechanism.contacts[1].index.duals[5:end] # remove the first 4 elements γ, ψ, β
    idx_slacks = mechanism.contacts[1].index.slacks[5:end] # remove the first 4 elements sγ, sψ, sβ
    options = mechanism.solver.options
    @show size(mechanism.solver.solution.all[idx_primals])
    @show size(floor_contact.detector.solver.solution.primals)
    mechanism.solver.solution.all[idx_primals] .= floor_contact.detector.solver.solution.primals
    mechanism.solver.solution.all[idx_duals] .= floor_contact.detector.solver.solution.duals .- options.complementarity_backstep
    mechanism.solver.solution.all[idx_slacks] .= floor_contact.detector.solver.solution.slacks .- options.complementarity_backstep

    Mehrotra.solve!(mechanism.solver)
    update_nodes!(mechanism)
    z1 = get_next_state(mechanism)
    return z1
end

################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-25.0]
z0 = [xp2; vp15]

u0 = zeros(3)
H0 = 300

@elapsed storage = simulate!(mech, z0, H0)


################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 10)

visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
# plot!(hcat(storage.variables...)')


function contact_initializer!(parent_pose, child_pose, detector::CollisionDetector{T,D,NC,NP}) where {T,D,NC,NP}
    detection = detector.detection
    solver = detector.solver
    solver.options.differentiate = true

    detection.parent_pose .= parent_pose
    detection.child_pose .= child_pose
    θ = get_parameters(detection)
    solver.parameters .= θ

    Mehrotra.solve!(solver)
    return nothing
end

# parent_pose = get_next_pose(mech.solver.solution.all, mech.bodies[1])
#
#
# mech.bodies[1]
# floor_contact = bilevel_mech.contacts[1]
#

# mech.contacts
# c, α, βp, βc, γ, ψ, β, λα, λp, λc, sγ, sψ, sβ, sα, sp, sc =
#     unpack_variables(x[contact.index.variables], contact)
# bilevel_mech.contacts
# c, α, βp, βc, λα, λp, λc, sα, sp, sc = unpack_variables(solver.solution.all, detection)

# pp3 = [0,1,0.0]
# contact_initializer(pp3, zeros(3), floor_contact.detector)
#
# floor_contact.detector.solver.solution.primals
# floor_contact.detector.solver.solution.duals
# floor_contact.detector.solver.solution.slacks
#
# idx_primals = mech.contacts[1].index.primals
# idx_duals = mech.contacts[1].index.duals[5:end]
# idx_slacks = mech.contacts[1].index.slacks[5:end]
# mech.solver.solution.all[idx_primals] .= floor_contact.detector.solver.solution.primals
# mech.solver.solution.all[idx_duals] .= floor_contact.detector.solver.solution.duals
# mech.solver.solution.all[idx_slacks] .= floor_contact.detector.solver.solution.slacks
#
# floor_contact.detector.solver.solution.duals
# floor_contact.detector.solver.solution.slacks
#
# floor_contact
