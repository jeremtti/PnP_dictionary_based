# Extended Experiments and Fast Algorithm for PnP with Synthesis-Based Denoiser

This repository extends [dictionary-based_denoisers_FBPnP](https://github.com/tomMoral/dictionary-based_denoisers_FBPnP), the companion repository for the paper:

**Analysis and Synthesis Denoisers for Forward-Backward Plug-and-Play Algorithms**  
[hal.science/hal-04786802](https://hal.science/hal-04786802)

---

## Installation

To install the required dependencies, install the packages listed in `requirements.txt`.

To download the BSDS500 dataset needed for experiments, run the script `create_dataset.py`.

---

## Overview of Experiments

All experiments are conducted on two tasks:
- **Deblurring**
- **Inpainting**

Each task (denoted `task ∈ {deblurring, inpainting}`) has two objectives:
1. **Study the convergence behavior of the algorithm**
2. **Obtain optimal reconstruction quality**

Useful functions are provided in the script:  
`experiments_{task}.py`

---

## I. Convergence Study

Goal: Analyze how the algorithm behaves for various values of:
- Inner iterations `L`
- Regularization parameter `λ`

Also includes a comparison between the **original** and **fast** versions of the algorithm.

Main script:  
`convergence_{task}.py`

Configuration file:  
`config_convergence_{task}.yaml`

Results are saved in the folder:  
`/convergence_{task}`

To generate plots of the results, use:  
`plot_convergence.py`  
(Note: `plot_convergence.py` réalise les plots correspondants.)

---

## II. Reconstruction Quality

### Optimal λ Search

This experiment selects the best regularization parameter `λ` using a warm-restart strategy, decreasing from `λ_max` to `0`.

Main script:  
`optimal_lambda_{task}.py`

Configuration file:  
`config_optimal_lambda_{task}.yaml`

To generate plots, use:  
`plot_optimal_lambda.py`

---

### Debiasing Analysis

This experiment studies:
- The effect of debiasing as a post-processing step
- Comparisons between **$l_1$-based** and **$l_2$-based** reconstructions

Notebook:  
`debiaising_{task}.ipynb`

Results are saved in the folder:  
`/images_{task}`
