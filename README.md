This is an index to show modules making up the dynamic lung MRI simulation pipeline.

The order of directories to visit is `simulator-data`, then either `simulations` or `simulations-mr0`.
Each of these directories has a README explaining project structure, virtual environments and what to run in which order. Each script listed as needing to be executed has a `--help` CLI flag to understand inputs.

The `simulation-experiments` directory contains code to run an example dynamic simulation on both KomaMRI and MRzeroCore, using a simple disk phantom, and compare results.
