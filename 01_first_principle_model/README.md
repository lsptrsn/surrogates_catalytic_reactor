# Model for creating reactor dynamics
This project is authored by Luisa Peterson. It focuses on dynamic modeling of a tubular methanation reactor based on [1].

## Getting started?
To get started with this project, follow the steps below:
1. Go to the `reactor_model` directory and run the `__main__.py` script. This script generates a surrogate model of the reactor dynamics using GNN and also performs the evaluation.

## Dependencies
The dependencies can be found in `environment_FPM.yml`. In addition, Jax needs to be installed manually as it depends on the plattform it is used on. For running this code, `jax 0.4.30` was used with `GPU (NVIDIA, CUDA 12, x86_64)`.
More information on installing Jax can be found here: `https://jax.readthedocs.io/en/latest/installation.html`

## Runtime
On the given computing system, the code runs within ~ 150 s.

## Questions?
If you have any questions or need further clarification, feel free to reach out to me. You can send an email to `peterson@mpi-magdeburg.mpg.de`.

## Literature
[1] Zimmermann, R. T., Bremer, J., & Sundmacher, K. (2022). Load-flexible fixed-bed reactors by multi-period design optimization. Chemical Engineering Journal, 428, 130771.
