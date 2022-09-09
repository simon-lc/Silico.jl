using GLVisualizer
using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using FileIO

vis = Visualizer()
open(vis)

include("../src/DojoLight.jl")

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.1

################################################################################
# GLVisualizer point-cloud
################################################################################
resolution = (301, 700)
glvis = GLVisualizer.Visualizer(resolution=resolution)
open(glvis)
GLVisualizer.set_camera!(glvis;
    eyeposition=[0,+2,3.0],
    lookat=[0,0,0.0],
    up=[0,1,0.0])


bunny_mesh = FileIO.load(joinpath(@__DIR__, "..", "deps", "bunny.obj"))
scaling = 7.0
GLVisualizer.setobject!(glvis, :root, :pbody, bunny_mesh; color=RGBA(0.3, 0.3, 0.3, 0.7))
GLVisualizer.scale!(glvis.scene[:pbody], scaling*ones(3)...)
q0 = GLVisualizer.axes_pair_to_quaternion([0, 0, -1.0], [0, 1.0, 0])
q1 = GLVisualizer.axes_pair_to_quaternion([1.0, 0, 0.0], [0, 1.0, 0])
GLVisualizer.settransform!(glvis, :pbody, [0, 0, -0.25], q1 * q0)

GLVisualizer.set_floor!(glvis)

p1 = [Int(floor((resolution[1]+1)/2))] # perfectly centered for uneven number of pixels
p2 = Vector(1:30:resolution[2])
n1 = length(p1)
n2 = length(p2)

include("../cvxnet/softmax.jl")
include("../cvxnet/loss.jl")
include("../cvxnet/point_cloud.jl")

eyeposition = [0,0,3.0]
lookat = [0,0,0.0]
up = [0,1,0.0]
E = [2*[0, cos(α), sin(α)] for α in range(0.1π, 0.9π, 5)]
WC = [point_cloud(glvis, p1, p2, resolution, e, lookat, up) for e in E]
for i = 1:5
    for j = 1:n1*n2
        setobject!(
            vis[:poincloud][Symbol(i)][Symbol(j)],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.025),
            MeshPhongMaterial(color=RGBA(0.1,0.1,0.1,1)))
        settransform!(vis[:poincloud][Symbol(i)][Symbol(j)], MeshCat.Translation(WC[i][:,j]...))
    end
end

plt = plot()
for i = 1:5
    plot!(plt, [norm(WC[i][2:3,j]) for j=1:n1*n2])
end
display(plt)

δ0 = 1e+2
Δ0 = 2e-2
βb0 = 1e-2
βo0 = 1e-9
βf0 = 1e-3
βμ0 = 1e-3
βc0 = 1e-2

n = 5
polytope_dimensions = [n,n,n,n,n,n]
nb = length(polytope_dimensions)
local_loss(θ) = loss(WC, E, θ, polytope_dimensions, βb=βb0, βo=βo0, βf=βf0, βμ=βμ0, βc=βc0, Δ=Δ0, δ=δ0)
local_initial_invH(θ) = zeros(nb*(2n+2),nb*(2n+2)) + 1e-3*I

θinit = zeros(0)
for i = 1:nb
    θi = [range(-π, π, length=n+1)[1:end-1]; 0.10*ones(n); zeros(2)] + [0.2*rand(2n); [0; 0.5rand(1)]]
    A, b, o = unpack_halfspaces(θi)
    # b .+= A * [0.2*(-1+2i/nb), 0.5]
    push!(θinit, pack_halfspaces(A, b, o)...)
end
θinit
Ainit, binit, oinit = unpack_halfspaces(θinit, polytope_dimensions)
@time local_loss(θinit)

@elapsed res = Optim.optimize(local_loss, θinit,
    # BFGS(),
    # Newton(),
    BFGS(initial_invH = local_initial_invH),
    autodiff=:forward,
    Optim.Options(
        # f_tol=1e-3,
        allow_f_increases=true,
        iterations=100,
        # callback=local_callback,
        extended_trace = true,
        store_trace = true,
        show_trace = false,
        # show_trace = true,
    ))

################################################################################
# unpack results
################################################################################
θopt = Optim.minimizer(res)
Aopt, bopt, oopt = unpack_halfspaces(θopt, polytope_dimensions)
solution_trace = [iterate.metadata["x"] for iterate in Optim.trace(res)]
plot(hcat(solution_trace...)', legend=false)

################################################################################
# visualize results
################################################################################
for i = 1:nb
    build_2d_polytope!(vis[:initial], Ainit[i], binit[i] + Ainit[i]*oinit[i], name=Symbol(i),
    # build_2d_polytope!(vis[:initial], Ainit[i], binit[i], name=Symbol(i),
            color=RGBA(1,0.3,0.0,0.5))
end

for j = 1:length(solution_trace)
    for i = 1:nb
        Aopt, bopt, oopt = unpack_halfspaces(solution_trace[j], polytope_dimensions)
        setobject!(
            vis[:optimized][Symbol(i)][Symbol(j)][:centroid],
            HyperSphere(MeshCat.Point(0, oopt[i]...), 0.040),
            MeshPhongMaterial(color=RGBA(1,0.1,0.1,1)))
        try
            build_2d_polytope!(vis[:optimized][Symbol(i)], Aopt[i], bopt[i] + Aopt[i]*oopt[i], name=Symbol(j),
            # build_2d_polytope!(vis[:optimized][Symbol(i)], Aopt[i], bopt[i], name=Symbol(j),
                color=RGBA(1,1,0.0,0.5))
        catch e
        end
    end
end

anim = MeshCat.Animation(20)
for j = 1:length(solution_trace)
    atframe(anim, j) do
        for jj = 1:length(solution_trace)
            for i = 1:nb
                setvisible!(vis[:optimized][Symbol(i)][Symbol(jj)], j==jj)
            end
        end
    end
end
MeshCat.setanimation!(vis, anim)

settransform!(vis[:reference], MeshCat.Translation(+0.0, 0,0))
for i = 1:nb
    settransform!(vis[:initial][Symbol(i)], MeshCat.Translation(+0.1i, 0,0))
end
for i = 1:nb
    settransform!(vis[:optimized][Symbol(i)], MeshCat.Translation(+0.1(nb+i), 0,0))
end

# vis = Visualizer()
# open(vis)
set_floor!(vis, x=1.0, origin=[-0.5,0,0])
set_light!(vis, direction="Negative", ambient=0.7)
set_background!(vis)

RobotVisualizer.set_camera!(vis, zoom=40.0)
