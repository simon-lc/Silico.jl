[![CI](https://github.com/simon-lc/DojoLight.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/simon-lc/DojoLight.jl/actions/workflows/CI.yml)

# DojoLight.jl
Lightweight and hackable version of the Dojo physics engine.

### CI on a private repo depending on another private repo
- create a private and a public key using 
`ssh-keygen -C "git@github.com:simon-lc/Mehrotra.jl.git"`
- put the public key into the DEPLOY KEYS of `Mehrotra.jl`
- put the private key into the secrets (action secrets) of DojoLight.jl
- use the `CI.yml` script given at the end of this thread:
https://discourse.julialang.org/t/github-ci-with-private-repository-in-julia/81207/10
