# Extended Experiments and Fast Algorithm for PnP with Synthesis-Based Denoiser

This repository extends [dictionary-based_denoisers_FBPnP](https://github.com/tomMoral/dictionary-based_denoisers_FBPnP), the companion repository for **Analysis and Synthesis Denoisers for Forward-Backward Plug-and-Play Algorithms** ([hal.science/hal-04786802](https://hal.science/hal-04786802)).

## Installation

To install the required dependencies, install the packages listed in `requirements.txt`.

To download the BSDS500 dataset needed for experiments, run the script `create_dataset.py`.

## Overview of Experiments
The experiments have two objectives:
1. **Study the convergence behavior of the algorithm**
2. **Obtain optimal reconstruction quality**

All experiments are conducted on a task of **deblurring** and a task of **inpainting**.
In the following, we denote  task \in {deblurring, inpainting}.

Useful functions are provided in the scripts `experiments_{task}.py`

## I. Convergence Study

The scripts `convergence_{task}.py` and their configurations `config_convergence_{task}.yaml` analyze the convergence of the algorithm for various values of inner iterations $L$ and regularization parameters $\lambda$ and compare the original and the fast versions of the algorithm.

The plots are generated with `plot_convergence.py`.

Results are saved in the folders `/convergence_{task}`

## II. Reconstruction Quality

The scripts `optimal_lambda_{task}.py` and their configurations `config_optimal_lambda_{task}.yaml` select the best regularization parameter $\lambda$ using a warm-restart strategy, decreasing from `λ_max` to `0`.

The plots are generated with `plot_optimal_lambda.py`

The notebooks `debiaising_{task}.ipynb` show the effect of debiaising as a post-processing step and compare the $l_1$-based and $l_2$-based reconstructions.

Results are saved in the folders `/images_{task}`
