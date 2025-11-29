# Simple test to check if basic FMM components work
function test_fmm_components()
    N = 100000
    positions = rand(3, N)
    momenta = zeros(3, N)
    beam = Particles(; pos=positions, mom=momenta, charge=-1.0, mass=1.0)
    
    println("=== Testing FMM Components ===")
    
    # Test ClusterTree
    println("1. Testing ClusterTree...")
    ct = ClusterTree(beam; N0=10, stretch=SVector(1.0,1.0,1.0))
    println("   ClusterTree created: parindices = $(length(ct.parindices))")
    println("   Clusters: $(length(ct.clusters.parlohis))")
    
    # Test if we can create MacroParticles
    println("2. Testing MacroParticles...")
    mp = MacroParticles(ct.clusters, 4)
    println("   MacroParticles created")
    
    println("3. Testing InteractionLists...")
    itlists = InteractionLists(ct.clusters; stretch=SVector(1.0,1.0,1.0), eta=0.5)
    println("   InteractionLists created: $(itlists.nm2l) M2L, $(itlists.np2p) P2P")
    
    println("=== Component Test Complete ===")
end

# Run the test
test_fmm_components()