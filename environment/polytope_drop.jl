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

    Af = [0.0  +1.0]
    bf = [0.0]
    normalize_A!(A)

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
