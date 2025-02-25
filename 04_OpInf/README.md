# OpInf for reactor dynamics
This project is authored by Luisa Peterson. It focuses on the application of Operator Inference (OpInf) for modeling reactor dynamics.

## Getting started?
To get started with this project, follow the steps below:

1. Insert the orginial data from `ieee_paper/code/02_FOM_data` in in the following folder:`ieee_paper/code/04_OpInf/IEEE_data`
2. Run the `opinf_methanation.py` script. This script generates a surrogate model of the reactor dynamics using OpInf.
3. The output data from the script is automatically stored in the `IEEE_eval` directory.
4. To reproduce the evaluation results, navigate to the `IEEE_eval` directory and run the `opinf_eval.py` script with the scenario of your interest.

## Runtime
On the given computing system, the code runs within ~ 3500 s.

## Questions?
If you have any questions or need further clarification, feel free to reach out to me. You can send an email to `peterson@mpi-magdeburg.mpg.de`.