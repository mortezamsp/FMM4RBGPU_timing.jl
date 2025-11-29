# More flexible particle structure
struct Particles{T}
    positions::AbstractArray{SVector{3,T},1}
    momenta::AbstractArray{SVector{3,T},1}
    efields::AbstractArray{SVector{3,T},1}
    bfields::AbstractArray{SVector{3,T},1}
    charge::T
    mass::T
    npar::Int
end

function Particles(; pos, mom, charge, mass)
    T = eltype(pos)
    npar = size(pos, 2)
    
    # Convert 2D arrays to 1D arrays of SVectors
    positions = [SVector{3,T}(pos[:, i]) for i in 1:npar]
    momenta = [SVector{3,T}(mom[:, i]) for i in 1:npar]
    efields = zeros(SVector{3,T}, npar)
    bfields = zeros(SVector{3,T}, npar)
    
    return Particles(positions, momenta, efields, bfields, charge, mass, npar)
end

# type alias for Particles
const Beam{T} = Particles{T}