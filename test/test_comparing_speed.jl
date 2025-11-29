using Revise
using FMM4RBGPU_timing
using CUDA
using Dates  # Import the Dates module

# Number of particles
const N = 32000

# Create position and momentum distribution of N particles
positions = rand(3, N)
momenta = zeros(3, N)

# Create a particle beam in which each particle's charge and mass are -1 and 1 
beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0) 

# Update particle field by FMM with the following parameters: 
#   n: degree of interpolation
#   N0: maximum number of particles in the leaf cluster
#   eta: admissibility parameter 
#   lambda: a characteristic length for the normalization of length quantity
const n = 4 
const N0 = 125  
const eta = 0.5

# Measure execution time
start_time = Dates.now()
update_particles_field!(beam, FMMGPU(eta=eta, N0=N0, n=n); lambda=1.0)
end_time = Dates.now()
gpu_time = end_time - start_time
println("FMMGPU time: $gpu_time")  # Corrected variable name to gpu_time


start_time = Dates.now()
update_particles_field!(beam, FMM(eta=eta, N0=N0, n=n); lambda=1.0)
end_time = Dates.now()
gpu_time = end_time - start_time
println("FMMGPU time: $gpu_time")  # Corrected variable name to gpu_time

