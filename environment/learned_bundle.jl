function get_learned_bundle(
    A::Vector{<:Matrix},
    b::Vector{<:Vector};
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    options=Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    nb = length(b)
    Af = [0.0  +1.0]
    bf = [0.0]

    # nodes
    parent_shapes = [PolytopeShape(A[i], b[i]) for i = 1:nb]
    bodies = [
        Body(timestep, mass, inertia, parent_shapes, gravity=+gravity, name=:pbody),
        ]
    contacts = [
        PolyHalfSpace(bodies[1], Af, bf,
            parent_collider_id=i,
            friction_coefficient=friction_coefficient,
            name=Symbol(i))
        for i = 1:nb]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    initialize_solver!(mechanism.solver)
    return mechanism
end
