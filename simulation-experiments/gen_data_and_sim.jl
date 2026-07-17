using NPZ, KomaMRI, PlotlyJS, Plots, Printf, CUDA
# include("RRVariability.jl")
include("../simulations/src/utils.jl")

sys, sim_params = setup_system("../simulations/scanner_properties.jsonc");

RRs          = [1.0]       # [s] constant RR interval
N_matrix     = 32          # image size = N x N
N_phases     = 16          # Number of cardiac phases
FOV          = 0.10        # [m]
TR           = 25e-3       # [s]
flip_angle   = 10          # [º]
adc_duration = 0.2e-3;     # [s]

obj = heart_phantom(; spins_per_voxel=49); # Needs to be odd to be symmetric around origin.
seq = read_seq("radial_2d.seq")
# seq = read_seq("/home/brock/biospig4-drive/simulator-data/sequences/aztek_radial_v4_compat_120_180_180_px_10s_runtime_2s_dummy_part_spoke.seq")
cds = stack([stack(get_spin_coords(obj.motion, obj.x, obj.y, obj.z, coord_t)) for coord_t in range(0, 1, 64)]) # Sample movement into 64 phases.
cds = permutedims(cds, (3, 1, 2)) # Fix shape

npzwrite("ph_for_mrzero.mrzphantom.npz", Dict("PD_map" => obj.ρ, "T1_map" => obj.T1, "T2_map" => obj.T2, "T2dash_map" => obj.T2s, "x" => obj.x, "y" => obj.y, "z" => obj.z, "deformations" => cds, "time_scale_max" => 1.0, "time_scale_min" => 0.0, "FOV" => [maximum(obj.x) - minimum(obj.x), maximum(obj.y) - minimum(obj.y), maximum(obj.z) - minimum(obj.z)]))

sig = simulate(obj, seq, sys; sim_params);
npzwrite("sig_koma.npy", sig)

sleep(5)

obj.motion = NoMotion()
sig_nomotion = simulate(obj, seq, sys; sim_params);
npzwrite("sig_koma_nomotion.npy", sig_nomotion)
