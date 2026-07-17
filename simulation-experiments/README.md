To reproduce results:

1. First clone the `simulations` and `simulations-mr0` directories, and install the required dependencies (Julia project, mr0-venv, recon-venv-gpunufft).
2. Source the mr0 venv you just installed.
3. Run `python generate_2d_sequence.py`
4. Run `julia --threads=auto --project=<PATH_TO_SIMULATIONS DIR> gen_data_and_sim.jl`
5. Run `python load_data_and_sim.py` and `python load_data_and_sim.py -c`
6. Source recon-venv-gpunufft
7. Run `marimo edit reconstruct_simulation_output.marimo.py`. Plots will be saved as SVG files in the `plots` directory.
