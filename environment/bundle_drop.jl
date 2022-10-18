function get_bundle_drop(;
        timestep=0.05,
        gravity=-9.81,
        mass=1.0,
        inertia=0.2 * ones(1,1),
        friction_coefficient=0.9,
        A=[
        [
            +1.0 -0.2;
            +0.0 +1.0;
            -1.0 -0.2;
            +0.0 -1.0;
            ],
        [
            +1.0 +0.3;
            +0.0 +1.0;
            -1.0 +0.1;
            +0.0 -1.0;
            ],
        [
            +1.0 -0.3;
            +0.0 +1.0;
            -1.0 +0.5;
            +0.0 -1.0;
            ],
        ],
        b=[
            0.5*[+1.5, +1.0, +1.5, +0.0],
            0.5*[+0.5, +2.0, +1.0, -1.0],
            0.5*[+0.8, +3.0, +0.0, -1.0],
        ],
        method_type::Symbol=:finite_difference,
        options=Mehrotra.Options(
            # verbose=false,
            complementarity_tolerance=1e-4,
            compressed_search_direction=false,
            max_iterations=30,
            sparse_solver=false,
            warm_start=true,
            )
        )

    np = length(A)
    normalize_A!.(A)

    # nodes
    parent_shapes = [PolytopeShape(A[i], b[i]) for i=1:np]
    bodies = [
        Body(timestep, mass, inertia, parent_shapes, gravity=gravity, name=:pbody),
        ]
    normal = [0.0, 1.0]
    position_offset = [0.0, 0.0]
    floor_shape = HalfspaceShape(normal, position_offset)
    contacts = [
        PolyHalfSpace(bodies[1], floor_shape,
            parent_shape_id=i,
            friction_coefficient=friction_coefficient,
            name=Symbol(:halfspace_p,i)) for i = 1:np]
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
