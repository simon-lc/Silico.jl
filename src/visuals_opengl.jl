######################################################################
# polytope
######################################################################
function build_polytope!(vis::GLVisualizer.Visualizer, parent::Symbol, name::Symbol,
        A::Matrix{T}, b::Vector{T};
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    h = hrep(A, b)
    p = polyhedron(h)
    m = Polyhedra.Mesh(p)
    GLVisualizer.setobject!(vis, parent, name, m, color=color)
    return nothing
end

function build_2d_polytope!(vis::GLVisualizer.Visualizer, parent::Symbol, name::Symbol,
        A::Matrix{T}, b::Vector{T};
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    n = size(A)[1]
    Ae = [zeros(n) A]
    Ae = [Ae;
         -1 0 0;
          1 0 0]
    be = [b; 0.05; 0.05]
    build_polytope!(vis, parent, name, Ae, be, color=color)
    return nothing
end

function build_2d_convex_bundle!(vis::GLVisualizer.Visualizer, parent::Symbol,
		θ::Vector, polytope_dimensions::Vector{Int},
		color=RGBA(0.8, 0.8, 0.8, 1.0))

	n = length(polytope_dimensions)
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)

	for i = 1:n
		name = Symbol(:polytope_, i)
		delete!(vis, name)
		build_2d_polytope!(vis, parent, name,
				A[i], b[i];
		        # A[i], b[i] + A[i] * o[i];
		        color=color)
	end
	return nothing
end


######################################################################
# shape
######################################################################
function build_shape!(vis::GLVisualizer.Visualizer, parent::Symbol, name::Symbol,
        shape::PolytopeShape1170; collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where T

    A = shape.A
    b = shape.b
	o = shape.o
    build_2d_polytope!(vis, parent, name, A, b + A * o, color=collider_color)
    return nothing
end

function build_shape!(vis::GLVisualizer.Visualizer, parent::Symbol, name::Symbol,
        shape::SphereShape1170; collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        ) where T

    GLVisualizer.setobject!(vis, parent, name,
        HyperSphere(GeometryBasics.Point(0,0,0.), shape.radius[1]),
        color=collider_color)
    GLVisualizer.setobject!(vis, parent, Symbol(name, :_fake),
        HyperSphere(GeometryBasics.Point(0,0,0.0005), shape.radius[1]),
        color=RGBA(0.8, 0.8, 0.8, 0.8))
    return nothing
end

######################################################################
# body
######################################################################
function build_body!(vis::GLVisualizer.Visualizer, body::Body;
        name::Symbol=body.name,
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        center_of_mass_color=RGBA(1, 1, 1, 1.0),
        center_of_mass_radius=0.025,
        ) where T
    # set parent invisible objet
    GLVisualizer.setobject!(vis, :root, name,
        HyperSphere(GeometryBasics.Point(0,0,0.), 0.0));

    # shapes
    for (i,shape) in enumerate(body.shapes)
        build_shape!(vis, name, Symbol(name, i), shape,
            collider_color=collider_color)
    end

    # center of mass
    GLVisualizer.setobject!(vis, name, Symbol(name, :_com),
        HyperSphere(GeometryBasics.Point(0,0,0.), center_of_mass_radius),
        color=center_of_mass_color);
    return nothing
end

function set_body!(vis::GLVisualizer.Visualizer, body::Body, pose; name=body.name)
    p = pose[1:2]
    q = pose[3:3]
    pe = [0; p]
    qe = RotX(q[1])
    qe = GLVisualizer.Quaternion(qe.v1, qe.v2, qe.v3, qe.s)
    GLVisualizer.settransform!(vis, name, pe, qe)
    return nothing
end

######################################################################
# mechanism
######################################################################
function build_mechanism!(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170;
        color=RGBA(0.2, 0.2, 0.2, 0.8))
    for body in mechanism.bodies
        build_body!(vis, body, collider_color=color)
    end
    return nothing
end

function set_mechanism!(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170, storage::Storage116, i::Int)
    for (j,body) in enumerate(mechanism.bodies)
        set_body!(vis, body, storage.x[i][j])
    end
    return nothing
end

function set_mechanism!(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170, z)
    off = 0
    for (j,body) in enumerate(mechanism.bodies)
        nz = state_dimension(body)
        set_body!(vis, body, z[off .+ (1:nz)]); off += nz
    end
    return nothing
end

function Base.delete!(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170)
    for body in mechanism.bodies
        delete!(vis, body.name)
    end
    return nothing
end
