function quaternion_product(q, p)
    qw = q[1]
    qv = q[2:4]
    pw = p[1]
    pv = p[2:4]

    qp = [qw * pw - qv' * pv; qw .* pv + pw .* qv + cross(qv, pv)]
    return qp
end

function dagger(q)
    qd = [q[1]; -q[2:4]]
    return qd
end

function quaternion_difference(q0, q1)
    return quaternion_product(dagger(q0), q1)
end

function quaternion_ϕ(q0, q1)
    ϕ = quaternion_product(dagger(q0), q1)[2:4]
end

function quaternion_increment(q, ϕ)
    quaternion_product(q, [sqrt(1 - ϕ'*ϕ); ϕ])
end

function Lmat(q)
    [
        q[1] -q[2] -q[3] -q[4];
        q[2]  q[1] -q[4]  q[3];
        q[3]  q[4]  q[1] -q[2];
        q[4] -q[3]  q[2]  q[1];
    ]
end

function Rmat(q)
    [
        q[1] -q[2] -q[3] -q[4];
        q[2]  q[1]  q[4] -q[3];
        q[3] -q[4]  q[1]  q[2];
        q[4]  q[3] -q[2]  q[1];
    ]
end

function Vmat()
    [
        0 1 0 0.0;
        0 0 1 0.0;
        0 0 0 1.0;
    ]
end

vector_rotate(q, x) = Vmat() * Lmat(q) * Rmat(q)' * Vmat()' * x



q0 = normalize([1.0, 1, 0, 0])
q1 = normalize([1.05, 1.0, 0, 0])

quaternion_product(q0, q1)
dagger(q0)
quaternion_difference(q0, q1)
ϕ01 = quaternion_ϕ(q0, q1)
quaternion_increment(q0, ϕ01) - q1

function get_3d_body_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    options=Mehrotra.Options(
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    # nodes
    shapes = [SphereShape(0.3,zeros(3))]
    bodies = [
        Body(timestep, mass, inertia, shapes, gravity=+gravity, name=:pbody, D=3),
        ]
    normal = [0.0, 0.0, 1.0]
    position_offset = [0.0, 0.0, 0.0]
    floor_shape = HalfspaceShape(normal, position_offset)
    contacts = [
        SphereHalfSpace(bodies[1], floor_shape;
            name=:floor,
            friction_coefficient=friction_coefficient)
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




# using Symbolics
using Plots
using Statistics
using Random

include("body.jl")

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
mass = 0.2;
inertia = 0.8 * Matrix(Diagonal(ones(3)));
friction_coefficient = 1.0

mech = get_3d_body_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Mehrotra.Options(
        verbose=false,
        complementarity_tolerance=1e-5,
        compressed_search_direction=true,
        # compressed_search_direction=false,
        sparse_solver=false,
        warm_start=true,
        )
    )

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 =  [+0.00, +0.00, +1.00, 1,0,0,0]
vp15 = [+0.00, -0.50, +1.50, +10,+10,0]
z0 = [xp2; vp15]

u0 = zeros(6)
H0 = 200

@elapsed storage = simulate!(mech, deepcopy(z0), H0)

################################################################################
# visualization
################################################################################
build_mechanism!(vis, mech)
set_mechanism!(vis, mech, storage, 1)

visualize!(vis, mech, storage, build=false)

scatter(storage.iterations)
plot(hcat(storage.variables...)')
mech.dimensions

variable_dimension.(mech.bodies)
variable_dimension.(mech.contacts)
storage



# origins = [0, 0, 0.0]
# widths = [0.1, 0.2, 0.3]
# obj = HyperRectangle(GeometryBasics.Vec(origins...), GeometryBasics.Vec(widths...))
# setobject!(vis[:fake], obj, MeshPhongMaterial(color=RGBA(0.8, 0.8, 0.8, 0.8)))
#
# q = Quaternion(1.0, 0.0, 0.0, 0.0)
# settransform!(vis[:fake], MeshCat.compose(
#     MeshCat.Translation(SVector{3}(0, 0, 0.0)),
#     MeshCat.LinearMap(rotationmatrix(q)),
#     )
# )
