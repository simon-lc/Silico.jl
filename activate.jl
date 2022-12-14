function module_dir()
    @__DIR__
end

using Pkg
Pkg.activate(module_dir())
using Silico

# using Pkg
# Pkg.add(url="https://github.com/simon-lc/Mehrotra.jl")
