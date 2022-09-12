################################################################################
# residual
################################################################################
function lp_contact_residual(primals, duals, slacks, parameters; np::Int=0, nc::Int=0, d::Int=0)
    xp, xc, Ap, bp, Ac, bc = unpack_lp_contact_parameters(parameters, np=np, nc=nc, d=d)

    y, z, s = primals, duals, slacks
    zp = z[1:np]
    zc = z[np .+ (1:nc)]
    sp = s[1:np]
    sc = s[np .+ (1:nc)]

    # pw is expressed in world's frame
    pw = y[1:d] + (xp[1:2] + xc[1:2]) ./ 2
    ϕ = y[d .+ (1:1)]
    # pp is expressed in pbody's frame
    pp = x_2d_rotation(xp[3:3])' * (pw - xp[1:2])
    # pc is expressed in cbody's frame
    pc = x_2d_rotation(xc[3:3])' * (pw - xc[1:2])

    res = [
        x_2d_rotation(xp[3:3]) * Ap' * zp + x_2d_rotation(xc[3:3]) * Ac' * zc;
        1 - sum(zp) - sum(zc);
        sp - (- Ap * pp + bp + ϕ .* ones(np));
        sc - (- Ac * pc + bc + ϕ .* ones(nc));
        # sp .* zp;
        # sc .* zc;
    ]
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
function unpack_lp_contact_parameters(parameters; np=0, nc=0, d=0)
    off = 0
    xp = parameters[off .+ (1:d+1)]; off += d+1
    xc = parameters[off .+ (1:d+1)]; off += d+1

    Ap = parameters[off .+ (1:np*d)]; off += np*d
    Ap = reshape(Ap, (np,d))
    bp = parameters[off .+ (1:np)]; off += np

    Ac = parameters[off .+ (1:nc*d)]; off += nc*d
    Ac = reshape(Ac, (nc,d))
    bc = parameters[off .+ (1:nc)]; off += nc

    return xp, xc, Ap, bp, Ac, bc
end

function pack_lp_contact_parameters(xp, xc, Ap, bp, Ac, bc)
    return [xp; xc; vec(Ap); bp; vec(Ac); bc]
end

################################################################################
# solver
################################################################################
function lp_contact_solver(Ap, bp, Ac, bc; d::Int=2,
        options::Options=Options(
            complementarity_tolerance=3e-3,
            residual_tolerance=1e-6,
            differentiate=true,
            compressed_search_direction=true,
            ))

    np = length(bp)
    nc = length(bc)

    xp2 = zeros(d+1)
    xc2 = zeros(d+1)

    parameters = pack_lp_contact_parameters(xp2, xc2, Ap, bp, Ac, bc)
    num_primals = d + 1
    num_cone = np + nc
    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    sized_residual(primals, duals, slacks, parameters) =
        lp_contact_residual(primals, duals, slacks, parameters; np=np, nc=nc, d=d)

    solver = Solver(
            sized_residual,
            num_primals,
            num_cone,
            parameters=parameters,
            nonnegative_indices=idx_nn,
            second_order_indices=idx_soc,
            options=options,
            )
    return solver
end


################################################################################
# utils
################################################################################
function set_pose_parameters!(solver::Solver, xp, xc)
    d = length(xp) - 1
    solver.parameters[1:d+1] .= xp
    solver.parameters[d+1 .+ (1:d+1)] .= xc
    return nothing
end

function set_parameters!(solver::Solver, xp, xc, Ap, bp, Ac, bc)
    solver.parameters .= pack_lp_contact_parameters(xp, xc, Ap, bp, Ac, bc)
    return nothing
end


################################################################################
# ContactSolver
################################################################################
struct ContactSolver{T,S,NP,NC,FO,Fϕ,FP,FC,FN}
    solver::S

    ϕ::Vector{T}
    x::Vector{T}
    N::Matrix{T}
    P::Matrix{T}
    V::Matrix{T}
    ∂ϕ::Matrix{T}
    ∂x::Matrix{T}
    nw::Vector{T}
    tw::Vector{T}
    outvariables::Vector{T}

    get_outvariables::FO
    get_ϕ::Fϕ
    get_p_parent::FP
    get_p_child::FC
    get_N::FN

    num_parameters::Int
    num_outvariables::Int
end

function ContactSolver(Ap::Matrix{T}, bp::Vector{T}, Ac::Matrix{T}, bc::Vector{T}; d::Int=2,
        checkbounds=true,
        threads=false,
        options::Options=Options(
            verbose=false,
            complementarity_tolerance=3e-3,
            residual_tolerance=1e-6,
            differentiate=true,
            compressed_search_direction=true)) where T

    @assert d == 2
    np = size(Ap, 1)
    nc = size(Ac, 1)
    nx = d + 1

    solver = lp_contact_solver(Ap, bp, Ac, bc; d=d, options=options)
    num_parameters = solver.dimensions.parameters
    num_variables = solver.dimensions.variables

    solution = Symbolics.variables(:solution, 1:num_variables)
    solution_sensitivity = Symbolics.variables(:solution_sensitivity,
        1:num_variables, 1:num_parameters)

    xl = get_outvariables(solution, solution_sensitivity, d=d, merged=true)
    num_outvariables = length(xl)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = get_outvariables(
        solution, solution_sensitivity, d=d, merged=false)

    xl_expr = Symbolics.build_function(xl, solution, solution_sensitivity,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    ϕ_expr = Symbolics.build_function(ϕ, solution, solution_sensitivity,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    p_parent_expr = Symbolics.build_function(p_parent, solution, solution_sensitivity,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    p_child_expr = Symbolics.build_function(p_child, solution, solution_sensitivity,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    N_child_expr = Symbolics.build_function(N, solution, solution_sensitivity,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    exprs = [xl_expr; ϕ_expr; p_parent_expr; p_child_expr; N_child_expr]

    return ContactSolver{T, typeof(solver), np, nc, typeof.(exprs)...}(
        solver,
        zeros(1),#ϕ
        zeros(d),#x
        zeros(1,6),#N
        zeros(1,6),#P
        zeros(1,6),#V
        zeros(1,6),#∂ϕ
        zeros(1,6),#∂x
        zeros(d),#nw
        zeros(d),#tw
        zeros(num_outvariables),
        exprs...,
        num_parameters,
        num_outvariables)
end

function get_outvariables(solver::Solver{T}; d::Int=2, merged::Bool=true) where T
    get_outvariables(solver.solution.all, solver.data.solution_sensitivity, merged=merged)
end

function get_outvariables(solution::Vector, solution_sensitivity::Matrix; d::Int=2, merged::Bool=true)

    p_parent = solution[1:d]
    p_child = solution[1:d]
    ϕ = solution[d .+ (1:1)]
    N = solution_sensitivity[d .+ (1:1), 1:2d+2]
    ∂p_parent = solution_sensitivity[1:d, :]
    ∂p_child = solution_sensitivity[1:d, :]
    nw = N[1:d] / norm(N[1:d])
    R = [0 1; -1 0]
    tw = R * nw

    merged && return [ϕ; p_parent; p_child; vec(N); vec(∂p_parent); vec(∂p_child); nw; tw]
    return ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child, nw, tw
end



function update_outvariables!(contact_solver::ContactSolver, parameters::Vector{T}) where T
    update_outvariables!(
        contact_solver.outvariables,
        parameters,
        contact_solver.solver,
        contact_solver.get_outvariables)
end

function update_outvariables!(outvariables::Vector{T}, parameters::Vector{T},
        solver::S, get_outvariables::F) where {T,S,F}

    solver.parameters .= parameters
    solve!(solver)
    get_outvariables(outvariables, solver.solution.all, solver.data.solution_sensitivity)
    return nothing
end





# Ap = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .- 0.10ones(4,2)
# bp = 0.5*[
#     +1,
#     +1,
#     +1,
#      2,
#     ]
# Ac = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .+ 0.10ones(4,2)
# bc = 0.5*[
#      1,
#      1,
#      1,
#      1,
#     ]
#
# contact_solver = ContactSolver(Ap, bp, Ac, bc)
# outvariables = rand(contact_solver.num_outvariables)
# parameters = deepcopy(contact_solver.solver.parameters)
#
# update_outvariables!(outvariables, parameters, contact_solver)
# get_outvariables(contact_solver.solver, merged=false)


# update_subvariables!(subvariables, subparameters, contact_solver)
# Main.@code_warntype update_subvariables!(subvariables, subparameters, contact_solver)
# @benchmark $update_subvariables!($subvariables, $subparameters, $contact_solver)
#



#
# ################################################################################
# # demo
# ################################################################################
# vis = Visualizer()
# render(vis)
# set_floor!(vis)
# set_light!(vis)
# set_bpckground!(vis)
#
# Ap = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .- 0.10ones(4,2)
# bp = 0.5*[
#     +1,
#     +1,
#     +1,
#      2,
#     ]
#
# Ac = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .+ 0.10ones(4,2)
# bc = 0.5*[
#      1,
#      1,
#      1,
#      1,
#     ]
# np = length(bp)
# nc = length(bc)
# d = 2
#
# build_2d_polyhedron!(vis, Ap, bp, color=RGBA(0.2,0.2,0.2,0.6), npme=:polya)
# build_2d_polyhedron!(vis, Ac, bc, color=RGBA(0.8,0.8,0.8,0.6), npme=:polyb)
#
# xp2 = [0.4,3.0]
# xc2 = [0,4.0]
# qa2 = [+0.5]
# qb2 = [-0.5]
#
# set_2d_polyhedron!(vis, xp2, qa2, npme=:polya)
# set_2d_polyhedron!(vis, xc2, qb2, npme=:polyb)
#
# contact_solver = lp_contact_solver(Ap, bp, Ac, bc; d=2,
#     options=Options(verbose=true, compressed_search_direction=true, differentiate=true))
# set_pose_parameters!(contact_solver, xp2, qa2, xc2, qb2, np=np, nc=nc, d=d)
#
# solve!(contact_solver)
# # @benchmark $solve!($contact_solver)
# # Main.@profiler [solve!(contact_solver) for i=1:1000]
#
#
# function search_direction!(solver::Solver; compressed::Bool=false)
#     dimensions = solver.dimensions
#     linear_solver = solver.linear_solver
#     data = solver.data
#     residual = data.residual
#     step = data.step
#
#     if compressed
#         step = compressed_search_direction!(linear_solver, dimensions, data, residual, step)
#     else
#         step = uncompressed_search_direction!(linear_solver, dimensions, data, residual, step)
#     end
#     return step
# end
#
# contact_solver_c = lp_contact_solver(Ap, bp, Ac, bc; d=2,
#     options=Options(verbose=true, differentiate=true, compressed_search_direction=true))
# contact_solver_u = lp_contact_solver(Ap, bp, Ac, bc; d=2,
#     options=Options(verbose=true, differentiate=true, compressed_search_direction=false))
#
#
# solve!(contact_solver_c)
# solve!(contact_solver_u)
# S0 = contact_solver_c.data.solution_sensitivity
# S1 = contact_solver_u.data.solution_sensitivity
#
# indices = contact_solver_c.indices
# norm(S0[indices.equality, :])
# norm(S1[indices.equality, :])
# norm(S0[indices.equality, :] - S1[indices.equality, :], Inf)
#
#
# duals = rand(8)
# slacks = rand(8)
#
# initialize_primals!(contact_solver_u)
# initialize_duals!(contact_solver_u)
# initialize_slacks!(contact_solver_u)
# contact_solver_u.solution.duals .= duals
# contact_solver_u.solution.slacks .= slacks
# evaluate!(contact_solver_u.problem,
#     contact_solver_u.methods,
#     contact_solver_u.cone_methods,
#     contact_solver_u.solution,
#     contact_solver_u.parameters,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     cone_constraint=true,
#     cone_jacobian=true,
#     cone_jacobian_inverse=true,
# )
#
# residual!(contact_solver_u.data,
#     contact_solver_u.problem,
#     contact_solver_u.indices,
#     contact_solver_u.solution,
#     contact_solver_u.parameters,
#     contact_solver_u.central_paths.zero_central_path, compressed=false)
# differentiate!(contact_solver_u)
# ustep = deepcopy(search_direction!(contact_solver_u, compressed=false))
#
#
# initialize_primals!(contact_solver_c)
# initialize_duals!(contact_solver_c)
# initialize_slacks!(contact_solver_c)
# contact_solver_c.solution.duals .= duals
# contact_solver_c.solution.slacks .= slacks
# evaluate!(contact_solver_c.problem,
#     contact_solver_c.methods,
#     contact_solver_c.cone_methods,
#     contact_solver_c.solution,
#     contact_solver_c.parameters,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     cone_constraint=true,
#     cone_jacobian=true,
#     cone_jacobian_inverse=true,
# )
#
# residual!(contact_solver_c.data,
#     contact_solver_c.problem,
#     contact_solver_c.indices,
#     contact_solver_c.solution,
#     contact_solver_c.parameters,
#     contact_solver_c.central_paths.zero_central_path, compressed=true)
# differentiate!(contact_solver_c)
# cstep = deepcopy(search_direction!(contact_solver_c, compressed=true))
# norm(cstep - ustep, Inf)
#
# S0 = contact_solver_c.data.solution_sensitivity
# S1 = contact_solver_u.data.solution_sensitivity
#
# indices = contact_solver_c.indices
# norm(S0[indices.equality, :])
# norm(S1[indices.equality, :])
# norm(S0[indices.equality, :] - S1[indices.equality, :], Inf)
# norm(S0[indices.slacks, :] - S1[indices.slacks, :], Inf)
#
# S0[indices.slacks, :]
# S1[indices.slacks, :]
#
#
# plot(Gray.(abs.(S0[indices.slacks, :] - S1[indices.slacks, :])))
# S0[indices.slacks, :]
#
#
#
# p = contact_solver.solution.primals[1:d]
# ϕ = contact_solver.solution.primals[d+1]
#
# setobject!(vis[:contacta],
#     HyperSphere(GeometryBasics.Point(0, ((xp2+xc2)/2 .+ p)...), 0.05),
#     MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
#
# contact_bundle(contact_solver, xp2, qa2, xc2, qb2; np=np, nc=nc, d=d, differentiate=true)
#
#
#
#
# # ################################################################################
# # # contact bundle parameters
# # ################################################################################
# # function contact_bundle(xl, parameters, solver::Solver;
# #         np::Int=0, nc::Int=0, d::Int=0)
# #     # v = variables [xp, qa, xc, qb]
# #     # ϕ = signed distance function
# #     # N = ∂ϕ∂v jacobian
# #     # pa = contact point on body a in world coordinptes
# #     # pb = contact point on body b in world coordinptes
# #     # vpa = ∂pa∂v jacobian, derivative of the contact point location not attached to body a
# #     # vpb = ∂pb∂v jacobian, derivative of the contact point location not attached to body a
# #
# #     # solver.parameters .= parameters
# #     # solver.options.differentiate = true
# #     # solve!(solver)
# #
# #     # ϕ = solver.solution.primals[d .+ (1:1)]
# #     # pa = solver.solution.primals[1:d]
# #     # pb = solver.solution.primals[1:d]
# #     # N = solver.data.solution_sensitivity[d .+ (1:1), 1:2d+2]
# #     # vpa = solver.data.solution_sensitivity[1:d, 1:2d+2]
# #     # vpb = solver.data.solution_sensitivity[1:d, 1:2d+2]
# #     #
# #     # return xl .= pack_contact_bundle(ϕ, pa, pb, N, vpa, vpb)
# # end
#
# # function contact_bundle_jacobian(jac, parameters, solver::Solver;
# #         np::Int=0, nc::Int=0, d::Int=0)
# #     solver.parameters .= parameters
# #     solver.options.differentiate = true
# #     solve!(solver)
# #
# #     ϕ = solver.solution.primals[d .+ (1:1)]
# #     pa = solver.solution.primals[1:d]
# #     pb = solver.solution.primals[1:d]
# #     ∂N = solver.data.solution_sensitivity[d .+ (1:1), 1:2d+2]
# #     ∂pa = solver.data.solution_sensitivity[1:d, 1:2d+2]
# #     ∂pb = solver.data.solution_sensitivity[1:d, 1:2d+2]
# #
# #     # ∂xl∂θl = ∂subvariables / ∂subparameters
# #     ∂xl∂θl = solver.data.solution_sensitivity
# #     return pack_contact_bundle(ϕ, pa, pb, N, ∂pa, ∂pb)
# # end
# #
# #
# #
# #
# # ################################################################################
# # # contact bundle parameters
# # ################################################################################
# # function unpack_contact_bundle(parameters; d::Int=0)
# #     off = 0
# #     ϕ = parameters[off .+ (1:1)]; off += 1
# #     pa = parameters[off .+ (1:d)]; off += d
# #     pb = parameters[off .+ (1:d)]; off += d
# #     N = parameters[off .+ (1:1*(2d+2))]; off += 1*(2d+2)
# #     N = reshape(N, (1,2d+2))
# #     vpa = parameters[off .+ (1:d*(2d+2))]; off += d*(2d+2)
# #     vpa = reshape(vpa, (d,2d+2))
# #     vpb = parameters[off .+ (1:d*(2d+2))]; off += d*(2d+2)
# #     vpb = reshape(vpb, (d,2d+2))
# #     return ϕ, pa, pb, N, vpa, vpb
# # end
# #
# # function pack_contact_bundle(ϕ, pa, pb, N, vpa, vpb)
# #     return [ϕ; pa; pb; vec(N); vec(vpa); vec(vpb)]
# # end
# #
# #
