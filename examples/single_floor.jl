using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots

include("../contact_model/lp_2d.jl")
include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

vis = Visualizer()
render(vis)

################################################################################
# residual
################################################################################
function joint_residual(primals, duals, slacks, parameters; np::Int=0, nc::Int=0, d::Int=0)
    xp2, xc2, vp15, vc15, Ap, bp, Ac, bc = unpack_mechanism_parameters(parameters, np=np, nc=nc, d=d)
    up = zeros(3)
    uc = zeros(3)
    timestep = 0.05
    mass = 1.0
    inertia = 0.2
    gravity = -9.81
    friction_coefficient = [0.3]

    y, z, s = primals, duals, slacks
    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:4]
    γpf = z[5:5]
    ψpf = z[6:6]
    βpf = z[7:8]
    γcf = z[9:9]
    ψcf = z[10:10]
    βcf = z[11:12]
    zp = z[4 .+ (1:np)]
    zc = z[4+np .+ (1:nc)]
    zppf = z[4+np+nc .+ (1:np)]
    zcpf = z[4+np+nc+np .+ (1:nf)]
    zpcf = z[4+np+nc+np+nf .+ (1:nc)]
    zccf = z[4+np+nc+nf+nf+nc .+ (1:nf)]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:4]
    sγpf = s[5:5]
    sψpf = s[6:6]
    sβpf = s[7:8]
    sγcf = s[9:9]
    sψcf = s[10:10]
    sβcf = s[11:12]
    sp = s[4 .+ (1:np)]
    sc = s[4+np .+ (1:nc)]
    sppf = s[4+np+nc .+ (1:np)]
    scpf = s[4+np+nc+np .+ (1:nf)]
    spcf = s[4+np+nc+np+nf .+ (1:nc)]
    sccf = s[4+np+nc+nf+nf+nc .+ (1:nf)]


    # pw is expressed in world's frame
    off = 0
    vp25 = y[off .+ (1:d+1)]; off += d+1
    vc25 = y[off .+ (1:d+1)]; off += d+1
    xp3 = xp2 + timestep * vp25
    xc3 = xc2 + timestep * vc25
    xp1 = xp2 - timestep * vp15
    xc1 = xc2 - timestep * vc15
    pw = y[off .+ (1:d)] + (xp3[1:2] + xc3[1:2]) ./ 2; off += d
    ϕ = y[off .+ (1:1)]; off += 1
    pwpf = y[off .+ (1:d)] + (xp3[1:2] + [0,0]) ./ 2; off += d
    ϕpf = y[off .+ (1:1)]; off += 1
    pwcf = y[off .+ (1:d)] + (xc3[1:2] + [0,0]) ./ 2; off += d
    ϕcf = y[off .+ (1:1)]; off += 1


    # pp is expressed in pbody's frame
    pp = x_2d_rotation(xp3[3:3])' * (pw - xp3[1:2])
    # pc is expressed in cbody's frame
    pc = x_2d_rotation(xc3[3:3])' * (pw - xc3[1:2])
    # pp is expressed in pbody's frame
    pppf = x_2d_rotation(xp3[3:3])' * (pwpf - xp3[1:2])
    # pc is expressed in cbody's frame
    pcpf = x_2d_rotation(xc3[3:3])' * (pwpf - xc3[1:2])
    # pp is expressed in pbody's frame
    ppcf = x_2d_rotation(xp3[3:3])' * (pwcf - xp3[1:2])
    # pc is expressed in cbody's frame
    pccf = x_2d_rotation(xc3[3:3])' * (pwcf - xc3[1:2])


    # normal and tangent
    npw = -x_2d_rotation(xp3[3:3]) * Ap' * zp
    ncw = +x_2d_rotation(xc3[3:3]) * Ac' * zc
    npwpf = -x_2d_rotation(xp3[3:3]) * Ap' * zppf
    ncwpf = +x_2d_rotation(xc3[3:3]) * Af' * zcpf
    npwcf = -x_2d_rotation(xp3[3:3]) * Ac' * zpcf
    ncwcf = +x_2d_rotation(xc3[3:3]) * Af' * zccf
    # nw ./= norm(nw)
    R = [0 1; -1 0]
    tpw = R * npw
    tcw = R * ncw
    tpwpf = R * npwpf
    tcwpf = R * ncwpf
    tpwcf = R * npwcf
    tcwcf = R * ncwcf

    # In the world frame
    p3_parent = pw
    p3_child = pw
    p3_parentpf = pwpf
    p3_childpf = pwpf
    p3_parentcf = pwcf
    p3_childcf = pwcf

    # force at the contact point in the contact frame
    νc = [β[1] - β[2]; γ]
    νcpf = [βpf[1] - βpf[2]; γpf]
    νccf = [βcf[1] - βcf[2]; γcf]

    # rotation matrix from contact frame to world frame
    wRc_p = [tpw npw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc_c = [tcw ncw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc_ppf = [tpwpf npwpf] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc_cpf = [tcwpf ncwpf] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc_pcf = [tpwcf npwcf] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc_ccf = [tcwcf ncwcf] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    # force at the contact point in the world frame
    νpw = +wRc_p * νc # parent
    νcw = -wRc_c * νc # child
    νpwpf = +wRc_ppf * νcpf # parent
    νcwpf = -wRc_cpf * νcpf # child
    νpwcf = +wRc_pcf * νccf # parent
    νcwcf = -wRc_ccf * νccf # child
    # wrenches at the centers of masses
    τpw = (skew([p3_parent - xp3[1:2]; 0]) * [νpw; 0])[3:3]
    τcw = (skew([p3_child  - xc3[1:2]; 0]) * [νcw; 0])[3:3]
    w = [νpw; τpw; νcw; τcw]
    τpwpf = (skew([p3_parentpf - xp3[1:2]; 0]) * [νpwpf; 0])[3:3]
    τcwpf = (skew([p3_childpf  - xc3[1:2]; 0]) * [νcwpf; 0])[3:3]
    wpf = [νpwpf; τpwpf]#; νcwpf; τcwpf]
    τpwcf = (skew([p3_parentcf - xp3[1:2]; 0]) * [νpwcf; 0])[3:3]
    τcwcf = (skew([p3_childcf  - xc3[1:2]; 0]) * [νcwcf; 0])[3:3]
    wcf = [νpwcf; τpwcf]#; νcwcf; τcwcf]
    # mapping the force into the generalized coordinates (at the centers of masses and in the world frame)

    vptan = vp25[1:2] + (skew([xp3[1:2]-p3_parent; 0]) * [zeros(2); vp25[3]])[1:2]
    vptan = vptan'*tpw
    vctan = vc25[1:2] + (skew([xc3[1:2]-p3_child;  0]) * [zeros(2); vc25[3]])[1:2]
    vctan = vctan'*tcw
    vtan = vptan - vctan

    vptanpf = vp25[1:2] + (skew([xp3[1:2]-p3_parent; 0]) * [zeros(2); vp25[3]])[1:2]
    vptanpf = vptan'*tpw
    vctanpf = vc25[1:2] + (skew([xc3[1:2]-p3_child;  0]) * [zeros(2); vc25[3]])[1:2]
    vctanpf = vctan'*tcw
    vtanpf = vptanpf - vctanpf

    M = Diagonal([mass, mass, inertia])
    res = [
        M * (xp3 - 2xp2 + xp1)/timestep - timestep * M * [0, +1e+0*gravity, 0];
        1e2*M * (xc3 - 2xc2 + xc1)/timestep - timestep * M * [0, -9e-1*gravity, 0];
        x_2d_rotation(xp3[3:3]) * Ap' * zp + x_2d_rotation(xc3[3:3]) * Ac' * zc;
        1 - sum(zp) - sum(zc);
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+vtan; -vtan] + ψ[1]*ones(2));
        sγpf - ϕpf;
        sψpf - (friction_coefficient[1] * γpf - [sum(βpf)]);
        sβpf - ([+vtanpf; -vtanpf] + ψpf[1]*ones(2));
        sγcf - ϕcf;
        sψcf - (friction_coefficient[1] * γcf - [sum(βcf)]);
        sβcf - ([+vtancf; -vtancf] + ψcf[1]*ones(2));
        sp - (- Ap * pp + bp + ϕ .* ones(np));
        sc - (- Ac * pc + bc + ϕ .* ones(nc));
        sppf - (- Ap * pppf + bp + ϕpf .* ones(np));
        scpf - (- Af * pcpf + bf + ϕpf .* ones(nf));
        spcf - (- Ac * ppcf + bc + ϕcf .* ones(nc));
        sccf - (- Af * pccf + bf + ϕcf .* ones(nf));
    ]
    res[1:2d+2] .-= w # -V3' * ν
    res[1:2d+2] .-= [wpf; wcf] # -V3' * ν
    return res
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end


################################################################################
# parameters
################################################################################
function unpack_mechanism_parameters(parameters; np=0, nc=0, d=0)
    off = 0
    xp = parameters[off .+ (1:d+1)]; off += d+1
    xc = parameters[off .+ (1:d+1)]; off += d+1
    vp = parameters[off .+ (1:d+1)]; off += d+1
    vc = parameters[off .+ (1:d+1)]; off += d+1

    Ap = parameters[off .+ (1:np*d)]; off += np*d
    Ap = reshape(Ap, (np,d))
    bp = parameters[off .+ (1:np)]; off += np

    Ac = parameters[off .+ (1:nc*d)]; off += nc*d
    Ac = reshape(Ac, (nc,d))
    bc = parameters[off .+ (1:nc)]; off += nc

    return xp, xc, vp, vc, Ap, bp, Ac, bc
end

function pack_mechanism_parameters(xp, xc, vp, vc, Ap, bp, Ac, bc)
    return [xp; xc; vp; vc; vec(Ap); bp; vec(Ac); bc]
end


################################################################################
# demo
################################################################################
# parameters
Af = [0.0 -1.0;]
bf = [0.0]
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp = 0.2*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ]

timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)

np = length(bp)
nc = length(bc)
nf = length(bf)
d = 2

xp2 = [+20,+20,0.0]
xc2 = [-2,-2,0.0]
vp15 = zeros(d+1)
vc15 = zeros(d+1)

parameters = pack_joint_parameters(xp2, xc2, vp15, vc15, Ap, bp, Ac, bc)
num_primals = 3*(d + 1)
num_cone = 4 + np + nc
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

sized_residual(primals, duals, slacks, parameters) =
    joint_residual(primals, duals, slacks, parameters; np=np, nc=nc, d=d)

solver = Solver(
        sized_residual,
        num_primals,
        num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        method_type=:finite_difference,
        options=Options(
            verbose=true,
            # verbose=true,
            # warm_start=true,
            # verbose=false,
            # sparse_solver=true,
            # compressed_search_direction=true,
            # max_iterations=1,
            # complementarity_decoupling=true,
            complementarity_tolerance=1e-3),
        );

parameters = pack_joint_parameters(xp2, xc2, vp15, vc15, Ap, bp, Ac, bc)
solver.parameters .= parameters
initialize_primals!(solver)
initialize_duals!(solver)
initialize_slacks!(solver)
initialize_interior_point!(solver)
solve!(solver)
# @benchmark $solve!($solver)



################################################################################
# test simulation
################################################################################
# solver.options.verbose = false#true
solver.options.complementarity_tolerance = 1e-5
solver.options.residual_tolerance


Xp2 = [[+0.00,+2.5,+1.0]]
Xc2 = [[-0.00,+0.5,-0.2]]
Vp15 = [[-0.0, 0.0, +10.0]]
Vc15 = [[+0.0, 0.0, +00.0]]
Pw = []
Nw = []
Tw = []
iter = []
solutions = []

H = 45
Up = [zeros(3) for i=1:H]
Uc = [zeros(3) for i=1:H]
for i = 1:H
    parameters = pack_joint_parameters(Xp2[end], Xc2[end], Vp15[end], Vc15[end], Ap, bp, Ac, bc)
    solver.parameters .= parameters
    solve!(solver)
    y = solver.solution.primals

    off = 0
    vp25 = y[off .+ (1:d+1)]; off += d+1
    vc25 = y[off .+ (1:d+1)]; off += d+1
    xp3 = Xp2[end] + timestep * vp25
    xc3 = Xc2[end] + timestep * vc25
    pw = y[off .+ (1:d)] + (xp3[1:2] + xc3[1:2]) ./ 2; off += d
    ϕ = y[off .+ (1:1)]; off += 1

    zp = solver.solution.duals[4 .+ (1:np)]
    nw = -x_2d_rotation(xp3[3:3]) * Ap' * zp
    R = [0 1; -1 0]
    tw = R * nw

    push!(Vp15, vp25)
    push!(Vc15, vc25)
    push!(Xp2, Xp2[end] + timestep * vp25)
    push!(Xc2, Xc2[end] + timestep * vc25)

    push!(Pw, pw)
    push!(Nw, nw)
    push!(Tw, tw)
    push!(iter, solver.trace.iterations)
    push!(solutions, deepcopy(solver.solution.all))
end



# add regularization scheme on LP

scatter(iter)

################################################################################
# visualization
################################################################################
render(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

build_rope(vis; N=1, color=Colors.RGBA(0,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:normal)

build_rope(vis; N=1, color=Colors.RGBA(1,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:tangent)

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_straight_rope(vis, [0; Pw[i]], [0; Pw[i]+Nw[i]]; N=1, name=:normal)
        set_straight_rope(vis, [0; Pw[i]], [0; Pw[i]+Tw[i]]; N=1, name=:tangent)
        set_2d_polytope!(vis, Xp2[i][1:2], Xp2[i][3:3], name=:polya)
        set_2d_polytope!(vis, Xc2[i][1:2], Xc2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pw[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
# open(vis)
# convert_frames_to_video_and_gif("single_level_hard_offset")
# convert_frames_to_video_and_gif("single_level_hard_tilted")
# convert_frames_to_video_and_gif("single_level_hard")

ex = solver.data.jacobian_variables_dense
# plot(Gray.(abs.(ex)))
# plot(Gray.(abs.(ex - ex')))
# plot(Gray.(abs.(ex + ex')))
# plot(Gray.(1e3abs.(solver.data.jacobian_variables_dense)))

ex[1:6,1:6]
ex[7:9,7:9]

# scatter(solver.solution.all)
# scatter(solver.solution.primals)
# scatter(solver.solution.duals)
# scatter(solver.solution.slacks)

# plot(hcat(solutions...)', legend=false)