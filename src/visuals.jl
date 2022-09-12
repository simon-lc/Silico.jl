######################################################################
# polytope
######################################################################
function build_polytope!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polytope,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    h = hrep(A, b)
    p = polyhedron(h)
    m = Polyhedra.Mesh(p)
    try
        setobject!(vis[name], m, MeshPhongMaterial(color=color))
    catch e
    end
    return nothing
end

function build_2d_polytope!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polytope,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    n = size(A)[1]
    Ae = [zeros(n) A]
    Ae = [Ae;
         -1 0 0;
          1 0 0]
    be = [b; 0.1; 00]
    build_polytope!(vis, Ae, be, name=name, color=color)
    return nothing
end

function set_polytope!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polytope) where T

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(p...),
        MeshCat.LinearMap(z_rotation(q)),
        )
    )
    return nothing
end

function set_2d_polytope!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polytope) where T
    pe = [0; p]

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(pe)),
        MeshCat.LinearMap(rotationmatrix(RotX(q[1]))),
        )
    )
    return nothing
end

######################################################################
# shape
######################################################################
function build_shape!(vis::Visualizer, shape::PolytopeShape;
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where T

    A = shape.A
    b = shape.b
    o = shape.o
    build_2d_polytope!(vis, A, b + A * o, color=collider_color)
    return nothing
end

function build_shape!(vis::Visualizer, shape::SphereShape;
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where T

    setobject!(vis,
        HyperSphere(GeometryBasics.Point(0,0,0.), shape.radius[1]),
        MeshPhongMaterial(color=collider_color));
    setobject!(vis[:fake],
        HyperSphere(GeometryBasics.Point(0,0,0.0005), shape.radius[1]),
        MeshPhongMaterial(color=RGBA(0.8, 0.8, 0.8, 0.8)));
    return nothing
end

######################################################################
# body
######################################################################
function build_body!(vis::Visualizer, body::Body;
        name::Symbol=body.name,
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        center_of_mass_color=RGBA(1, 1, 1, 1.0),
        center_of_mass_radius=0.025,
        ) where T

    # shapes
    for (i,shape) in enumerate(body.shapes)
        build_shape!(vis[:bodies][name][Symbol(i)], shape, collider_color=collider_color)
    end

    # center of mass
    setobject!(vis[:bodies][name][:com],
        HyperSphere(GeometryBasics.Point(0,0,0.), center_of_mass_radius),
        MeshPhongMaterial(color=center_of_mass_color));
    return nothing
end

function set_body!(vis::Visualizer, body::Body, pose; name=body.name)
    p = pose[1:2]
    q = pose[3:3]
    pe = [0; p]
    settransform!(vis[:bodies][name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(pe)),
        MeshCat.LinearMap(rotationmatrix(RotX(q[1]))),
        )
    )
    return nothing
end

######################################################################
# 2D frame
######################################################################
function build_2d_frame!(vis::Visualizer;
    name::Symbol=:contact,
    origin_color=RGBA(0.2, 0.2, 0.2, 0.8),
    normal_axis_color=RGBA(0, 1, 0, 0.8),
    tangent_axis_color=RGBA(1, 0, 0, 0.8),
    origin_radius=0.025,
    ) where T

    # axes
    build_segment!(vis[:contacts][name];
        color=tangent_axis_color,
        segment_radius=origin_radius/2,
        name=:tangent)

    build_segment!(vis[:contacts][name];
        color=normal_axis_color,
        segment_radius=origin_radius/2,
        name=:normal)

    # origin
    setobject!(vis[:contacts][name][:origin],
        HyperSphere(GeometryBasics.Point(0,0,0.), origin_radius),
        MeshPhongMaterial(color=origin_color));
    return nothing
end

function set_2d_frame!(vis::Visualizer, contact, origin, normal, tangent; name=contact.name)
    settransform!(vis[:contacts][name][:origin],
        MeshCat.Translation(SVector{3}(0, origin...)))
    set_segment!(vis[:contacts][name], [0; origin], [0; origin+normal]; name=:normal)
    set_segment!(vis[:contacts][name], [0; origin], [0; origin+tangent]; name=:tangent)
    return nothing
end

######################################################################
# segment
######################################################################
function build_segment!(vis::Visualizer; color=Colors.RGBA(0,0,0,1),
        segment_radius=0.02, name=:segment)

    segment = MeshCat.Cylinder(MeshCat.Point(0,0,0.0), MeshCat.Point(0,0,1.0), segment_radius)
    material = MeshPhongMaterial(color=color)
    setobject!(vis[name][:scaled], segment, material)
    return vis
end

function set_segment!(vis::Visualizer, x_start, x_goal; name::Symbol=:segment)
    transform, scaling = RobotVisualizer.link_transform(x_start, x_goal)
    settransform!(vis[name][:scaled], scaling)
    settransform!(vis[name], transform)
    return nothing
end

######################################################################
# mechanism
######################################################################
function build_mechanism!(vis::Visualizer, mechanism::Mechanism;
        show_contact::Bool=true,
        color=RGBA(0.2, 0.2, 0.2, 0.8),
        center_of_mass_color=RGBA(1,1,1,1.0),
        name::Symbol=:robot)

    for body in mechanism.bodies
        build_body!(vis[name], body, collider_color=color, center_of_mass_color=center_of_mass_color)
    end
    if show_contact
        for contact in mechanism.contacts
            build_2d_frame!(vis[name], name=contact.name)
        end
    end
    return nothing
end

function set_mechanism!(vis::Visualizer, mechanism::Mechanism, storage::Storage116,
        i::Int; show_contact::Bool=true, name::Symbol=:robot)

    for (j,body) in enumerate(mechanism.bodies)
        set_body!(vis[name], body, storage.x[i][j])
    end
    if show_contact
        for (j, contact) in enumerate(mechanism.contacts)
            ii = max(1,i-1) # needed otherwise the contact frame is a one step ahead of the bodies.
            origin = storage.contact_point[ii][j]
            normal = storage.normal[ii][j]
            tangent = storage.tangent[ii][j]
            set_2d_frame!(vis[name], contact, origin, normal, tangent)
        end
    end
    return nothing
end

function set_mechanism!(vis::Visualizer, mechanism::Mechanism, z; name::Symbol=:robot)
    off = 0
    for (j,body) in enumerate(mechanism.bodies)
        nz = state_dimension(body)
        set_body!(vis[name], body, z[off .+ (1:nz)]); off += nz
    end
    return nothing
end

function visualize!(vis::Visualizer, mechanism::Mechanism, storage::Storage116{T,H};
        build::Bool=true,
        show_contact::Bool=true,
        name::Symbol=:robot,
        animation=MeshCat.Animation(Int(floor(1/mechanism.bodies[1].timestep[1])))) where {T,H}

    build && build_mechanism!(vis, mechanism, show_contact=show_contact, name=name)
    for i = 1:H
        atframe(animation, i) do
            set_mechanism!(vis, mechanism, storage, i, show_contact=show_contact, name=name)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end


function visualize!(vis::Visualizer, mechanism::Mechanism, z;
        build::Bool=true,
        name::Symbol=:robot,
        animation=MeshCat.Animation(Int(floor(1/mechanism.bodies[1].timestep[1])))) where {T}

    H = length(z)
    build && build_mechanism!(vis, mechanism, show_contact=false, name=name)
    for i = 1:H
        atframe(animation, i) do
            set_mechanism!(vis, mechanism, z[i], name=name)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end
