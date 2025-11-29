module FMM4RBGPU_timing
using LinearAlgebra
using StaticArrays
using CUDA
using Dates

export Particles, Beam

include("particles.jl")
include("utils/utils.jl")
include("cluster_tree/cluster_tree.jl")
include("fmm/fmm.jl")
include("gpu_fmm/gpu_fmm.jl")

# Import the complete implementation as a submodule
include("update_particles_field.jl")
using .FMM4RBGPUComplete: BruteForce, FMM, FMMGPU, update_particles_field!

end