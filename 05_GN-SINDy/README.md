# SINDy for reactor dynamics
This project is authored by Ali Forootani and adapted by Luisa Peterson. It focuses on the application of Sparse Identification of Nonlinear Dynamics (SINDy) for modeling reactor dynamics.

This code relies on the DeePyMoD package `https://github.com/PhIMaL/DeePyMoD`.

## Getting started?
To get started with this project, follow the steps below:
1. Insert the orginial data from `ieee_paper/code/02_FOM_data` in in the following folder:`ieee_paper/code/05_SINDy/data/IEEE_paper`
2. Change the directory to `05_SINDy/examples`, e.g. via `cd examples`.
3. Run the `examples/system_identification.py` script in the `examples` directory. This script generates a surrogate model of the reactor dynamics using SINDy.
4. The output data from the script is automatically stored in the `eval` directory.
5. To reproduce the evaluation results, navigate to the `eval` directory and run the `SINDy_eval.py` script.

## Runtime
On the given computing system, the `examples/system_identification.py` script runs within ~ 350 s.

## Remarks
The forward prediction with the complete data does not always work with computers with low memory capacity. We have therefore defined a variable “x” that only calls up every xth entry of the matrices and vectors. Increasing this number to 2 or 10 should provide a remedy and still deliver very accurate results.

## Questions?
If you have any questions or need further clarification, feel free to reach out to me. You can send an email to `forootani@mpi-magdeburg.mpg.de` or `peterson@mpi-magdeburg.mpg.de`.