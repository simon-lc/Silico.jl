

v1 = [+1,+1]
v2 = [-1,+1]
v3 = [-1,-1]
v4 = [+1,-1]


using Polyhedra
import GLPK

vrep(v1, v2, v3, v4)

poly = convexhull(v1, v2, v3, v4)
poly = hrep(poly)

x = [0,0,1,0.1]
y = [0,1,0,0.1]
v = vrep([x y])
v = removevredundancy(v, GLPK.Optimizer)
poly = Polyhedra.polyhedron(v)
hr = hrep(poly)
poly

hr.halfspaces[1].a
hr.halfspaces[1].β
fieldnames(typeof(hr.halfspaces[1]))

function vertex_to_halfspace(v::Vector)
    # v = nv x 2
    nv = Int(length(v)/2)
    v = reshape(v, (nv, 2))
    o = mean(v, dims=1)[1,:]
    vr = vrep(v)
    vr = removevredundancy(vr, GLPK.Optimizer)
    poly = Polyhedra.polyhedron(vr)
    hr = hrep(poly)
    nh = length(hr.halfspaces)
    A = zeros(nh, 2)
    b = zeros(nh)
    for (i,h) in enumerate(hr.halfspaces)
        A[i,:] .= h.a
        b[i] = h.β
    end
    b = b - A * o
    θ = [reshape(A, 2*nh); b; o]
    return θ, A, b, o
end

v = [x y]
vertex_to_halfspace(v)
@benchmark FiniteDiff.finite_difference_jacobian(x -> vertex_to_halfspace(x)[1], [x; y])
