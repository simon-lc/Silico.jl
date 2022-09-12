# solvers
include("solvers/adam.jl")
include("solvers/bfgs.jl")
include("solvers/newton_solver.jl")
include("solvers/sr1.jl")

# sys id
include("system_identification/visuals.jl")
include("system_identification/methods.jl")
include("system_identification/structure.jl")
include("system_identification/system_identification.jl")
include("system_identification/direct_system_identification.jl")

include("halfspace.jl")
include("softmax.jl")
include("transparency_point_cloud.jl")
include("utils.jl")
include("vertices.jl")
include("visuals.jl")
