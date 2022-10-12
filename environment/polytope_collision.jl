function get_polytope_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    A=[
    [
        +1.0 -0.0;
        +0.0 +1.0;
        -1.0 -0.0;
        +0.0 -1.0;
        ],
    [
        +1.0 +0.0;
        +0.0 +1.0;
        -1.0 +0.0;
        +0.0 -1.0;
        ],
    ],
    b=[
        0.5*[+1.0, +1.0, +1.0, +1.0],
        0.5*[+1.0, +1.0, +1.0, +1.0],
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

    N = length(A)
    # nodes
    shapes = [PolytopeShape(A[i], b[i]) for i=1:N]
    floor_shape = HalfspaceShape([0.0, 1.0])

    bodies = [Body(timestep, mass, inertia, shapes[i:i],
        gravity=+gravity, name=Symbol(:body_, i)) for i=1:N]

    body_contacts = vcat([
        [Contact2D(bodies[i], bodies[j],
            friction_coefficient=friction_coefficient,
            name=Symbol(:contact_, i, :_, j)) for i=1:j-1]
        for j=1:N]...)
    env_contacts = [
        EnvContact2D(bodies[i], floor_shape,
            friction_coefficient=friction_coefficient,
            name=Symbol(:env_contact_, i)) for i=1:N]

    contacts = [body_contacts; env_contacts]
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
