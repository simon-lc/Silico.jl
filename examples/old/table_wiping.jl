using Mehrotra
using MeshCat
using LinearAlgebra
using StaticArrays

include("../../examples/base/particle_utils.jl")

vis = Visualizer()
open(vis)
set_floor!(vis, z = 2.0, color=RGBA(0.5,0.5,0.5,0.5))
set_light!(vis)
set_background!(vis)

# dimensions and indices
num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]

# parameters
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = [0.4, 0.8, 0.9]
timestep = 0.10
mass = 1.0
gravity = 1.0*-9.81
friction_coefficient = 1.1
side = 0.5

parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]

################################################################################
# solve
################################################################################
solver = Solver(linear_particle_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        residual_tolerance=1e-8,
        complementarity_tolerance=1e-3,
        max_iterations=30,
        verbose=false,
        differentiate=true,
        warm_start=false)
    )
solve!(solver)

################################################################################
# simulation
################################################################################
H0 = 500
p2 = [1,1,1.0]
v15 = 0.0*[0,-1,1.0]
U = [-10*[0,0,1] for i=1:H0]
p, v, iterations = simulate_particle(solver, p2, v15, U;
    timestep=timestep,
    mass=mass,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side)

################################################################################
# visualization
################################################################################
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H0
    MeshCat.atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.Translation(SVector{3}(p[i])))
    end
end
MeshCat.setanimation!(vis, anim)



function dynamics(solver, z, u; timestep=0.10, mass=1.0, gravity=1.0*-9.81, friction_coefficient=1.1, side=0.5)
    p2 = z[1:3]
    v15 = z[4:6]
    solver.parameters .= [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]
    set_parameters!(solver, parameters)
    solve!(solver)

    v25 = solver.solution.primals
    γ = solver.solution.duals[1:1]
    p3 = p2 + timestep * v25
    z1 = [p3; v25; γ]

    v25dz = [solver.data.solution_sensitivity[solver.indices.primals, 1:6] zeros(3,1)]
    p3dz = [I(3) zeros(3,4)] + timestep * v25dz
    γdz = [solver.data.solution_sensitivity[solver.indices.duals[1:1], 1:6] zeros(1,1)]
    z1dz = [p3dz; v25dz; γdz]

    v25du = solver.data.solution_sensitivity[solver.indices.primals, 7:9]
    p3du = timestep * v25du
    γdu = solver.data.solution_sensitivity[solver.indices.duals[1:1], 7:9]
    z1du = [p3du; v25du; γdu]

    return z1, z1dz, z1du
end

p2 = [0, 0, 0.25]
v15 = [0, 0, 0.0]
z0 = [p2; v15]
u0 = [0, 0, 0.0]
dynamics(solver, z0, u0)
# @benchmark dynamics(solver, z0, u0)

function cost(z0, z, u, λ, zref, g, H;
        Qz=Diagonal([1e1,1e1,1e1, 1e-1,1e-1,1e-1, 1e0]),
        Qu=Diagonal([1e-4, 1e-4, 1e-3]),
        gradient::Bool=true,
        hessian::Bool=true,
        lagrangian::Bool=false,
        )
    N = length(z)
    c = 0
    gradient && (g .= 0.0)
    hessian && (H .= 0.0)
    violation = 0.0

    # compute cost
    off = 0
    zprev = copy(z0)
    for i = 1:N
        # indices
        indz = off .+ (1:7); off += 7
        indu = off .+ (1:3); off += 3
        indλ = off .+ (1:7); off += 7

        # dynamics
        zdyn, dzdyn, dudyn = dynamics(solver, zprev, u[i])
        zprev = copy(z[i])
        # control cost
        c += 0.5 * u[i]' * Qu * u[i]
        # state cost
        c += 0.5 * (z[i] - zref[i])' * Qz * (z[i] - zref[i])
        # constraint cost
        # @show size(λ[i])
        # @show size(zdyn)
        # @show size(z[i])
        lagrangian && (c += λ[i]' * (zdyn - z[i]))
        violation = max(violation, norm(zdyn - z[i], Inf))

        if gradient
            # gradient control
            g[indu] += Qu * u[i]
            # gradient state
            g[indz] += Qz * (z[i] - zref[i])

            # gradient constraints
            g[indz] += -λ[i]
            (i > 1) && (g[indz .- 17] += dzdyn' * λ[i])
            g[indu] += dudyn' * λ[i]
            g[indλ] += (zdyn - z[i])
        end

        if hessian
            # Hessian control
            H[indu, indu] += Qu
            # Hessian state
            H[indz, indz] += Qz
            # Hessian state
            H[indλ, indu] += dudyn
            H[indu, indλ] += dudyn'
            H[indλ, indz] += -I
            H[indz, indλ] += -I
            (i > 1) && (H[indλ, indz .- 17] += dzdyn)
            (i > 1) && (H[indz .- 17, indλ] += dzdyn')
        end
    end
    return c, g, H, violation
end

function newton_method(z0, z, u, λ, zc, uc, λc, zref, g, H; iterations=20)
    for i = 1:iterations
        c, g, H, violation = cost(z0, z, u, λ, zref, g, H, lagrangian=true)
        Δ = - H \ g

        α = 1.0
        for j = 1:3
            off = 0
            for k = 1:N
                # indices
                indz = off .+ (1:7); off += 7
                indu = off .+ (1:3); off += 3
                indλ = off .+ (1:7); off += 7

                zc[k] = z[k] + α * Δ[indz]
                uc[k] = u[k] + α * Δ[indu]
                λc[k] = λ[k] + α * Δ[indλ]
            end
            c_candidate, _, _, violation_candidate = cost(z0, zc, uc, λc, zref, g, H, gradient=false, hessian=false, lagrangian=true)
            ((c_candidate <= c)  || (violation_candidate <= violation)) && break
            α /= 2
        end
        @show α, c
        for k = 1:N
            z[k] .= zc[k]
            u[k] .= uc[k]
            λ[k] .= λc[k]
        end
    end
    return z, u, λ
end


function unpack(x, N::Int)
    z = [zeros(7) for i=1:N]
    u = [zeros(3) for i=1:N]
    λ = [zeros(7) for i=1:N]
    off = 0
    for i = 1:N
        # indices
        indz = off .+ (1:7); off += 7
        indu = off .+ (1:3); off += 3
        indλ = off .+ (1:7); off += 7
        z[i] .= x[indz]
        u[i] .= x[indu]
        λ[i] .= x[indλ]
    end
    return z, u, λ
end

function pack(z, u, λ)
    N = length(z)
    return vcat([vcat(z[i], u[i], λ[i]) for i = 1:N]...)
end


N = 10
H = spzeros(17N, 17N)
g = zeros(17N)

p2 = [0,0,0.25]
v15 = [0,0,0.0]
γ0 = [0.0]
z0 = [p2; v15; γ0]
z1 = [2, 2, 0.0, 0, 0, 0.0, 0.0]
u0 = [-0,-0,-0.0]



z = [copy(z0) for i=1:N]
u = [copy(u0) for i = 1:N]
λ = [zeros(7) for i = 1:N]

zc = [zeros(7) for i = 1:N]
uc = [zeros(3) for i = 1:N]
λc = [zeros(7) for i = 1:N]

zref = [z1 for i = 1:N]
c, g, H = cost(z0, z, u, λ, zref, g, H, lagrangian=true)

plot(g)
scatter((-H \ g)[1:17])

plot(Gray.(abs.(Matrix(H))))
@elapsed z, u, λ = newton_method(z0, z, u, λ, zc, uc, λc, zref, g, H, iterations=20)
0.05/20

plot(hcat(zref...)')
plot(hcat(z...)')
plt = plot(hcat(u...)'[:,1])
plot!(plt, hcat(u...)'[:,2])
plot!(plt, hcat(u...)'[:,3], linewidth=3)



# x = pack(z, u, λ)
# x = 1 .+ rand(17N)
# g0 = FiniteDiff.finite_difference_gradient(x -> cost(z0, unpack(x, N)..., zref, g, H, lagrangian=true)[1], x)
# H0 = FiniteDiff.finite_difference_hessian(x -> cost(z0, unpack(x, N)..., zref, g, H, lagrangian=true)[1], x)
# g1 = cost(z0, unpack(x, N)..., zref, g, H)[2]
# H1 = cost(z0, unpack(x, N)..., zref, g, H)[3]
# plot(g0 - g1)
# norm(g0 - g1)
# plot(H0 - H1)
# norm(H0 - H1)
# plot(Gray.(abs.(Matrix(H0))))
# plot(Gray.(100abs.(Matrix(H1 - H0))))




# z1, dz1, du1 = dynamics(solver, z0, u0)
# dz2 = FiniteDiff.finite_difference_jacobian(z -> dynamics(solver, z, u0)[1], z0)
# du2 = FiniteDiff.finite_difference_jacobian(u -> dynamics(solver, z0, u)[1], u0)
# norm(dz1 - dz2)
# norm(du1 - du2)

################################################################################
# visualization
################################################################################
ztraj = [z0, z...]
utraj = [zeros(3), u...]
setobject!(vis[:particle],
    HyperRectangle(-side/2 .* Vec(1,1,1), side .* Vec(1,1,1)),
    MeshPhongMaterial(color=RGBA(1,1,1,0.5)))
anim = MeshCat.Animation(Int(floor(1/timestep)))
build_segment!(vis, color=RGBA(0,0,0,1), name=:γ)
build_segment!(vis, color=RGBA(0,1,0,1), name=:input)

for i = 1:length(ztraj)
    MeshCat.atframe(anim, i) do
        p2 = ztraj[i][1:3]
        pc = p2 - [0, 0, side/2]
        γ = ztraj[i][7]
        settransform!(vis[:particle], MeshCat.Translation(SVector{3}(p2)))
        set_segment!(vis, pc, pc + 1*[0, 0, 1] * γ, name=:γ)
        set_segment!(vis, pc, pc + timestep*utraj[i], name=:input)
    end
end
MeshCat.setanimation!(vis, anim)

RobotVisualizer.convert_frames_to_video_and_gif("table_wiping_gamma_0")


min -0.5 * (x - x0)' * (x - x0)
s.t. (x - x1)' * Q * (x - x1) <= r1
s.t. (x - x1)' * Q * (x - x1) - r1 <= 0
s.t. s + (x - x1)' * Q * (x - x1) - r1 = 0

L = -0.5 * (x - x0)' * (x - x0) + γ' * (s + (x - x1)' * Q * (x - x1) - r1) - ρ Σ log(si)

kkt
∇x = -(x - x0) + γ' * 2(x - x1)
∇s = γ - ρ / s => γ .* s = ρ
∇γ = s + (x - x1)' * Q * (x - x1) - r1
bowl center = x0
object center = x1
object radius = r1
contact point = x
contact normal = x0 - x

function residual(primals, duals, slacks, parameters)

    res =

    return res
end
