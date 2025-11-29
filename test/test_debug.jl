using Revise
using FMM4RBGPU_timing
using CUDA
using Dates  

# Smaller test first
const N = 100000  # Start small
println("Testing with N = $N particles")

# Create position and momentum distribution of N particles
positions = rand(3, N)
momenta = zeros(3, N)

# Create a particle beam
beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0) 

const n = 4 
const N0 = 125  
const eta = 0.5

println("Testing CPU FMM...")
start_time = Dates.now()
result = update_particles_field!(beam, FMM(eta=eta, N0=N0, n=n); lambda=1.0)
end_time = Dates.now()
cpu_time = end_time - start_time
println("FMM time: $cpu_time")
println("FMM returned: $result")
println("Type of result: $(typeof(result))")

# Check if fields were actually computed
println("First few E-fields: $(beam.efields[1:3])")
println("First few B-fields: $(beam.bfields[1:3])")