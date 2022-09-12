function graph_symbolic_methods(symgraph::SymbolicNet, dim::Dimensions, ind::Indices;
        checkbounds=true,
        threads=false)

    parallel = threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parameter_keywords = ind.parameter_keywords

    idx_nn = ind.cone_nonnegative
    idx_soc = ind.cone_second_order

    x = Symbolics.variables(:x, 1:dim.variables) # variables
    θ = Symbolics.variables(:θ, 1:dim.parameters) # parameters
    κ = Symbolics.variables(:κ, 1:dim.cone) # central path
    r = Symbolics.variables(:r, 1:dim.variables) # residual
    rs = Symbolics.variables(:rs, 1:dim.cone) # cone product residual with corrections
    Δz = Symbolics.variables(:Δz, 1:dim.cone) # dual step
    Δza = Symbolics.variables(:Δza, 1:dim.cone) # dual affine step
    Δsa = Symbolics.variables(:Δsa, 1:dim.cone) # slack affine step
    y = x[ind.primals]
    z = x[ind.duals]
    s = x[ind.slacks]

    # equality residual
    f = dim.parameters > 0 ?
        func(y, z, s, θ) :
        func(y, z, s)

    # equality jacobians
    fx = Symbolics.sparsejacobian(f, x)

    # correction
    c = copy(r)
    c[ind.cone_product] .-= (κ - cone_product(Δza, Δsa, idx_nn, idx_soc))

    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        @warn warning
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end

    # correction compressed
    function correction_compressed(cc, r, Δza, Δsa, x, κ)
        error(warning)
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        error(warning)
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        # f(out, θ) = equality_constraint(out, x, θ)
        # matrix_cache = reshape(vector_cache, (dim.equality, dim.parameters))
        # FiniteDiff.finite_difference_jacobian!(matrix_cache, f, θ)
        # return nothing
        error(warning)
    end

    equality_jacobian_keywords = Vector{Function}()
    for k in eachindex(parameter_keywords)
        function func(vector_cache, x, θ)
            # function f(out, θi)
            #     θc = copy(θ)
            #     θc[parameter_keywords[k]] .= θi
            #     equality_constraint(out, x, θc)
            # end
            # matrix_cache = reshape(vector_cache, (dim.equality, length(parameter_keywords[k])))
            # FiniteDiff.finite_difference_jacobian!(matrix_cache, f, θ[parameter_keywords[k]])
            # return nothing
            error(warning)
        end
        push!(equality_jacobian_keywords, func)
    end

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))
    eθ_indices = zeros(Int, dim.equality, dim.parameters)
    vec(eθ_indices) .= 1:length(eθ_indices)
    ek_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        indices = vec(eθ_indices[:,val])
        push!(ek_indices, indices)
    end

    # # sparsity
    # fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    # fxc_sparsity = collect(zip([findnz(fxc)[1:2]...]...))
    # fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))
    # fθ_indices = similar(fθ, Int)
    # fθ_indices.nzval .= 1:nnz(fθ_indices)
    # fk_indices = Vector{Vector{Int}}()
    # for (key, val) in parameter_keywords
    #     idx = fθ_indices[:,val].nzval
    #     push!(fk_indices, idx)
    # end

    # expressions
    f_expr = Symbolics.build_function(f, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    c_expr = Symbolics.build_function(c, r, Δza, Δsa, κ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    methods = ProblemMethods(
        e_expr,
        equality_constraint_compressed,
        ex_expr,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        equality_jacobian_keywords,
        c_expr,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_indices,
    )

    return methods
end




function mechanism_symbolic_graph(residual, )


    return symgraph
end

using Graphs
using GraphRecipes
using Plots

include(joinpath(module_dir(), "SymbolicNet", "src", "macro.jl"))
include(joinpath(module_dir(), "SymbolicNet", "src", "net.jl"))
include(joinpath(module_dir(), "SymbolicNet", "src", "evaluation.jl"))



timestep=0.05
gravity=-9.81
mass=1.0
inertia=0.2 * ones(1,1)
friction_coefficient=0.9
Af = [0.0  +1.0]
bf = [0.0]
parent_shapes = [PolytopeShape(Af, bf),]
body = Body(timestep, mass, inertia, parent_shapes, gravity=+gravity, name=:pbody)
indexing!([body])

function residual!(e, x, θ, body::Body; symbolic_parsing=false)
    @rootlayer x
    @layer θ
    @layer e

    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25
    @layer p1
    @layer p3

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    optimality = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    @layer optimality
    eout = e
    # eout[index.optimality] .+= optimality
    eout[index.optimality] .+= optimality
    @leaflayer eout
    return nothing
end

function local_residual(e, x, θ; symbolic_parsing=false)
    return residual!(e, x, θ, body; symbolic_parsing=symbolic_parsing)
end

function get_name(dict::Dict, i::Int)
    for (key, val) in dict
        l = min(length(string(key)),3)
        (val == i) && (return string(key)[1:l])
    end
end

nθ = length(body.index.parameters)
nx = length(body.index.variables)

symgraph = generate_symgraph(local_residual, [nx, nx, nθ])
name_vector = [get_name(symgraph.name_dict, i) for i=1:nv(symgraph.graph)]
plt = graphplot(symgraph.graph, names=name_vector, curvature_scalar=0.01, linewidth=3, fontsize=15)


e0 = rand(nx)
x0 = rand(nx)
θ0 = rand(nθ)



evaluation50!(symgraph, [e0, x0, θ0])





function generate_gradients(func::Function, num_equality::Int, num_variables::Int,
        num_parameters::Int;
        checkbounds=true,
        threads=false)

    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)

    f .= e
    func(f, x, θ)

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity
end

abstract type NodeMethods{T,E,EX,Eθ} end

struct DynamicsMethods{T} <: AbstractProblemMethods{T,E,EX,EP}
    methods::Vector{NodeMethods}
    α::T
end

struct BodyMethods{T,E,EX,Eθ} <: NodeMethods{T,E,EX,Eθ}
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function BodyMethods(body::Body, dimensions::MechanismDimensions)
    r!(e, x, θ) = body_residual!(e, x, θ, body)
    f, fx, fθ, fx_sparsity, fθ_sparsity = generate_gradients(r!, dimensions.equality,
        dimensions.variables, dimensions.parameters)
    return BodyMethods(
        f,
        fx,
        fθ,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
        )
end

struct ContactMethods{T,E,EX,Eθ,C,S} <: NodeMethods{T,E,EX,Eθ}
    contact_solver::C
    subvariables::Vector{T}
    subparameters::Vector{T}

    set_subparameters!::S
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function ContactMethods(contact::PolyPoly, pbody::Body, cbody::Body,
        dimensions::MechanismDimensions;
        checkbounds=true,
        threads=false)


    contact_solver = ContactSolver(
        contact.A_parent_collider,
        contact.b_parent_collider,
        contact.A_child_collider,
        contact.b_child_collider,
        )

    num_equality = dimensions.equality
    num_variables = dimensions.variables
    num_parameters = dimensions.parameters
    num_subvariables = contact_solver.num_subvariables
    num_subparameters = contact_solver.num_subparameters
    subvariables = zeros(num_subvariables)
    subparameters = zeros(num_subparameters)

    # set_subparameters!
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)
    v25_parent = unpack_body_variables(x[pbody.index.x])
    v25_child = unpack_body_variables(x[cbody.index.x])

    x2_parent, _, _, timestep_parent = unpack_body_parameters(θ[pbody.index.θ], pbody)
    x2_child, _, _, timestep_child = unpack_body_parameters(θ[cbody.index.θ], cbody)
    x3_parent = x2_parent .+ timestep_parent[1] * v25_parent
    x3_child = x2_child .+ timestep_child[1] * v25_child
    @show x2_parent, timestep_parent, x3_parent, v25_parent

    Ap, bp, Ac, bc = unpack_contact_parameters(θ[contact.index.θ], contact)

    # θl = fct(x, θ)
    θl = [x3_parent; x3_child; vec(Ap); bp; vec(Ac); bc]
    θl2 = [x2_parent; x2_child; vec(Ap); bp; vec(Ac); bc]

    set_subparameters! = Symbolics.build_function(θl, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    # evaluation
    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    xl = Symbolics.variables(:xl, 1:num_subvariables)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl, contact)


    f .= e
    contact_residual!(f, x, xl, θ, contact, pbody, cbody)

    # for this one we are missing only third order tensors
    fx = Symbolics.sparsejacobian(f, x)
    fx[contact.index.e, [pbody.index.x; cbody.index.x]] .+= -sparse(N)
    # for this one we are missing ∂ϕ/∂θ and third order tensors
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, xl, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return ContactMethods(
        contact_solver,
        subvariables,
        subparameters,
        set_subparameters!,
        f_expr,
        fx_expr,
        fθ_expr,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
    )
end

function mechanism_methods(bodies::Vector, contacts::Vector, dimensions::MechanismDimensions)
    methods = Vector{NodeMethods}()

    # body
    for body in bodies
        push!(methods, BodyMethods(body, dimensions))
    end

    # contact
    for contact in contacts
        # TODO here we need to avoid hardcoding body1 and body2 as paretn and child
        push!(methods, ContactMethods(contact, bodies[1], bodies[2], dimensions))
    end

    return DynamicsMethods(methods, 1.0)
end

################################################################################
# evaluate
################################################################################

# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::Vector{NodeMethods}) where T
#     e .= 0.0
#     ex .= 0.0
#     eθ .= 0.0
#     for m in methods
#         evaluate!(e, ex, eθ, x, θ, m)
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::BodyMethods{T,E,EX,Eθ}) where {T,E,EX,Eθ}
#
#     methods.equality_constraint(e, e, x, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::ContactMethods{T,S}) where {T,S}
#
#     contact_solver = methods.contact_solver
#     xl = methods.subvariables
#     θl = methods.subparameters
#
#     # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
#     methods.set_subparameters!(θl, x, θ)
#     update_subvariables!(xl, θl, contact_solver)
#
#     # modify e, ex, eθ in-place using symbolics methods taking x, θ, xl as inputs
#     methods.equality_constraint(e, e, x, xl, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end

function evaluate!(
        problem::ProblemData{T},
        methods::DynamicsMethods{T},
        cone_methods::ConeMethods{T,B,BX,P,PX},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        ) where {T,B,BX,P,PX}

    # TODO this method allocates memory, need fix

    # reset
    problem.equality_constraint .= 0.0
    problem.equality_jacobian_variables .= 0.0
    problem.equality_jacobian_parameters .= 0.0

    # apply all methods
    for method in methods.methods
        evaluate!(problem, method, solution, parameters;
            equality_constraint=equality_constraint,
            equality_jacobian_variables=equality_jacobian_variables,
            equality_jacobian_parameters=equality_jacobian_parameters)
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
    )

    return nothing
end

function evaluate!(problem::ProblemData{T},
        methods::BodyMethods{T,E,EX,Eθ},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        ) where {T,E,EX,Eθ}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end

function evaluate!(problem::ProblemData{T},
        # methods::ContactMethods{T,E,EX,Eθ},
        methods::ContactMethods{T,S},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        # ) where {T,E,EX,Eθ}
        ) where {T,S}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
    contact_solver = methods.contact_solver
    xl = methods.subvariables
    θl = methods.subparameters
    methods.set_subparameters!(θl, x, θ)
    update_subvariables!(xl, θl, contact_solver)
    # @show x[1:3]
    # @show x[4:6]
    # @show θl[1:3]
    # @show θl[4:6]
    # @show xl[1]

    # update equality constraint and its jacobiens
    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, xl, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end
