# capse_paper

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2307.14339)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8187935.svg)](https://doi.org/10.5281/zenodo.8187935)

Repository containing the material for running the Capse.jl emulator and reproduce all the results of the paper.

In order to reproduce the results, you need to do two different things:

1. Clone this repository
2. Create a `data` folder in the repository
3. Download and unzip the two files contained in the Zenodo link above. If you care just about chains, you can download only the `chains_weights` data.
4. Install `julia` 1.9.2 on your system ( guide can be found [here](https://github.com/JuliaLang/juliaup)). Our codes do work also with other `julia` versions, but this is the version we put in the static environments we put in the notebooks folder: using it you will be able to run everythin without any problem.
5. Install [`IJulia`](https://github.com/JuliaLang/IJulia.jl) in order to have a `julia` kernel for the `jupyter notebooks`.
6. Open the notebooks folder, open the one you are interested it and run it! The notebooks should download and install the relevant packages in an automatic way.

We will update this page, in case, there are problems with the instructions reported here.

In case, contact any of us!

## Authors

Marco Bonici, PostDoctoral Researcher at INAF-IASF Milano,
Federico Bianchini, PostDoctoral researcher at Stanford
Jaime Ruiz-Zapatero, PhD Student at Oxford
