# Towards Digital Twins for Power-to-X: Comparing Surrogate Models for a Catalytic CO2 Methanation Reactor
by Luisa Peterson, Ali Forootani, Edgar Ivan Sanchez Medina, Ion Victor Gosea, Kai Sundmacher, and Peter Benner

# Contains
Python implementation to reproduce the results in [1]. The dependencies are given in each folder as different Python environments are used for all subroutines.
* `01_First principle model`: code to reproduce the mechanistic data
* `02_Full order model data`: data from the mechanistic model that is used for the surrogate models
* `03a_GNN`: code to build up the GNN surrogate model
* `03b_GNN+POD`: code to build up the GNN surrogate model on reduced data
* `04_OpInf`: code to build up the OpInf surrogate model
* `05_SINDy`: code to build up the SINDy surrogate model

# Runtime
We will report the runtime of each code inside the respective folder. To obtain the times, we used the same computer system with the characteristics given below.

**Computational Resources:**
- Processor: 12th Gen Intel Core i5-12600K
- Memory: 32.0 GB
- Graphics: NVIDIA Corporation/Mesa Intel Graphics (ADL-S GT1)
- OS: Ubuntu 20.04.6 LTS (64-bit)
- Storage: 1.3 TB

# License
See the [LICENSE](LICENSE) file for license rights and limitations (MIT).

# References
[1] Luisa Peterson, Ali Forootani, Edgar Ivan Sanchez Medina, et al. Towards Digital Twins for Power-to-X: Comparing Surrogate Models for a Catalytic CO2 Methanation Reactor. TechRxiv. August 02, 2024.
DOI: 10.36227/techrxiv.172263007.76668955/v1