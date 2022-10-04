# Config

gin configuration files for experiments.

To test a configuration in `main.py`, add `_test` to the end of a name, e.g.
`config/cma_me.gin_test`. Then, the original config (`config/cma_me.gin`) and
`config/test.gin` will be included.

`hpc/` contains configurations for HPC slurm nodes (see `scripts/run_slurm.sh`).
