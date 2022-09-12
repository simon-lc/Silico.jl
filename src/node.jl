abstract type Node{T} end
abstract type AbstractBody{T} <: Node{T} end

variable_dimension(node::Node) = primal_dimension(node) + 2 * cone_dimension(node)
optimality_dimension(node::Node) = primal_dimension(node)
slackness_dimension(node::Node) = cone_dimension(node)
equality_dimension(node::Node) = optimality_dimension(node) + slackness_dimension(node)

function find_body(bodies::AbstractVector{<:AbstractBody}, name::Symbol)
    idx = findfirst(x ->x == name, getfield.(bodies, :name))
    return bodies[idx]
end

mutable struct NodeIndices
    optimality::Vector{Int}
    slackness::Vector{Int}
    equality::Vector{Int}
    primals::Vector{Int}
    duals::Vector{Int}
    slacks::Vector{Int}
    variables::Vector{Int}
    parameters::Vector{Int}
end

function NodeIndices()
    return NodeIndices(
        collect(1:0),
        collect(1:0),
        collect(1:0),
        collect(1:0),
        collect(1:0),
        collect(1:0),
        collect(1:0),
        collect(1:0),
    )
end


function indexing!(nodes::Vector)
    # residual
    off = 0
    for node in nodes
        n = optimality_dimension(node)
        node.index.optimality = collect(off .+ (1:n)); off += n
    end
    for node in nodes
        n = slackness_dimension(node)
        node.index.slackness = collect(off .+ (1:n)); off += n
    end
    for node in nodes
        index = node.index
        index.equality = [index.optimality; index.slackness]
    end

    # variables
    off = 0
    for node in nodes
        n = primal_dimension(node)
        node.index.primals = collect(off .+ (1:n)); off += n
    end
    for node in nodes
        n = cone_dimension(node)
        node.index.duals = collect(off .+ (1:n)); off += n
    end
    for node in nodes
        n = cone_dimension(node)
        node.index.slacks = collect(off .+ (1:n)); off += n
    end
    for node in nodes
        index = node.index
        index.variables = [index.primals; index.duals; index.slacks]
    end

    # parameters
    off = 0
    for node in nodes
        n = parameter_dimension(node)
        node.index.parameters = collect(off .+ (1:n)); off += n
    end
    return nothing
end
