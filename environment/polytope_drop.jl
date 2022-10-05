function get_polytope_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    A = [
        +1.0 +0.2;
        +0.1 +1.0;
        -1.0 +0.3;
        +0.0 -1.0;
        ],
    b = 0.5*[
        +1,
        +1,
        +1,
        +1,
        ],
    options=Mehrotra.Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    for i = 1:length(b)
        A[i,:] ./= norm(A[i,:])
    end
    Af = [0.0  +1.0]
    bf = [0.0]

    # nodes
    shapes = [PolytopeShape(A, b)]
    bodies = [
        Body(timestep, mass, inertia, shapes, gravity=+gravity, name=:pbody),
        ]
    contacts = [
        PolyHalfSpace(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        ]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    Mehrotra.initialize_solver!(mechanism.solver)
    return mechanism
end

# function get_polytope_drop(;
#     timestep=0.05,
#     gravity=-9.81,
#     mass=1.0,
#     inertia=0.2 * ones(1,1),
#     friction_coefficient=0.9,
#     options=Options(
#         # verbose=false,
#         complementarity_tolerance=1e-4,
#         compressed_search_direction=false,
#         max_iterations=30,
#         sparse_solver=false,
#         warm_start=true,
#         )
#     )
#
#     Af = [0.0  +1.0]
#     bf = [0.0]
#     Ap1 = [
#         1.0  0.0;
#         0.0  1.0;
#         -1.0  0.0;
#         0.0 -1.0;
#         ] .- 0.30ones(4,2);
#     bp1 = 0.2*[
#         +1,
#         +1,
#         +1,
#         1,
#         ];
#     Ap2 = [
#         1.0  0.0;
#         0.0  1.0;
#         -1.0  0.0;
#         0.0 -1.0;
#         ] .+ 0.20ones(4,2);
#     bp2 = 0.2*[
#         -0.5,
#         +1,
#         +1.5,
#         1,
#         ];
#     Ac = [
#          1.0  0.0;
#          0.0  1.0;
#         -1.0  0.0;
#          0.0 -1.0;
#         ] .+ 0.20ones(4,2);
#     bc = 0.2*[
#         1,
#         1,
#         1,
#         1,
#         ];
#
#     # nodes
#     bodies = [
#         Body(timestep, mass, inertia, [Ap1, Ap2], [bp1, bp2], gravity=+gravity, name=:pbody),
#         Body(timestep, mass, inertia, [Ac], [bc], gravity=+gravity, name=:cbody),
#         ]
#     contacts = [
#         PolyPoly(bodies[1], bodies[2],
#             friction_coefficient=friction_coefficient,
#             name=:contact_1),
#         # PolyPoly(bodies[1], bodies[2],
#         #     parent_collider_id=2,
#         #     friction_coefficient=friction_coefficient,
#         #     name=:contact_2),
#         PolyHalfSpace(bodies[1], Af, bf,
#             friction_coefficient=friction_coefficient,
#             name=:halfspace_p1),
#         # PolyHalfSpace(bodies[1], Af, bf,
#         #     parent_collider_id=2,
#         #     friction_coefficient=friction_coefficient,
#         #     name=:halfspace_p2),
#         # PolyHalfSpace(bodies[2], Af, bf,
#         #     friction_coefficient=friction_coefficient,
#         #     name=:halfspace_c),
#         ]
#     indexing!([bodies; contacts])
#
#     return bodies, contacts
# end



# options=Options(
#         verbose=false,
#         complementarity_tolerance=1e-4,
#         # compressed_search_direction=true,
#         max_iterations=30,
#         sparse_solver=false,
#         warm_start=true,
#         )

# bodies, contacts = get_polytope_drop(;
#     timestep=0.05,
#     gravity=-9.81,
#     mass=1.0,
#     inertia=0.2 * ones(1,1),
#     friction_coefficient=0.9,
#     options=options,
#     )

# local_mechanism_residual(primals, duals, slacks, parameters) =
#     mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

# # mechanism = Mechanism(local_mechanism_residual, bodies, contacts, options=options)

# # # Dimensions
# nodes = [bodies; contacts]
# dim = MechanismDimensions(bodies, contacts)
# num_primals = sum(primal_dimension.(nodes))
# num_cone = sum(cone_dimension.(nodes))

# # indexing
# indexing!(nodes)

# # solver
# parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)

# # methods = mechanism_methods(bodies, contacts, dim)
# solver = Solver(
#         local_mechanism_residual,
#         num_primals,
#         num_cone,
#         parameters=parameters,
#         nonnegative_indices=collect(1:num_cone),
#         second_order_indices=[collect(1:0)],
#         # method_type=:finite_difference,
#         options=options
#         );

# Mehrotra.initialize_solver!(solver)
# solver.parameters .= rand(solver.dimensions.parameters)
# solve!(solver)
# solver.methods.equality_constraint
# @benchmark $solve!($solver)




# xp2 = [+0.0,1.5,-0.25]
# vp15 = [-0,0,-0.0]
# z0 = [xp2; vp15]
# u0 = zeros(3)

# set_current_state!(mechanism, z0)
# set_input!(mechanism, u0)
# update_parameters!(mechanism)
# solve!(mechanism.solver)


# mechanism.solver.dimensions




# # @benchmark $evaluate!($(solver.problem),
# #     $(solver.methods),
# #     $(solver.cone_methods),
# #     $(solver.solution),
# #     $(solver.parameters);
# #     equality_constraint=true,
# #     equality_jacobian_variables=true,
# #     equality_jacobian_parameters=true,
# #     cone_constraint=false,
# #     cone_jacobian=false,
# #     cone_jacobian_inverse=false,
# #     sparse_solver=false,
# #     )

# # evaluate!(solver.problem,
# solver.methods,
# solver.cone_methods,
# solver.solution,
# solver.parameters;
# equality_constraint=false,
# equality_jacobian_variables=false,
# equality_jacobian_parameters=false,
# cone_constraint=false,
# cone_jacobian=false,
# cone_jacobian_inverse=false,
# sparse_solver=false,
# )
