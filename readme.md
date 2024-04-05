# Locating and Editing Facts in Mamba

As SOTA network architectures continue to evolve, we need to ask if techniques developed for one type of architecture (transformer LMs) can also be applied to another (Mamba), and how much of the insights transfer across architectures.

In the context of factual recall, we investigate
1. If facts can be localized to particular modules and token positions in Mamba by adapting activation-patching / causal tracing to Mamba.

2. We check if we can edit facts with [ROME](https://rome.baulab.info/) in Mamba by directly editing one of the projection matrices in MambaBlock architecture.

3. We further investigate to the extent to which insights from works such as [Hernandez et al. (2023)](https://lre.baulab.info/) and [Geva et al. (2023)](https://arxiv.org/abs/2304.14767) generalize to Mamba. We identify that it is hard to implement techniques similar to attention knockout (used in [Geva et al. (2023)](https://arxiv.org/abs/2304.14767)) in Mamba due to certain architectural choices.

Checkout our [paper](https://arxiv.org/pdf/2404.03646.pdf) for details.

## Setup

All codes are written in Python >= 3.10 and tested in Ubuntu 20.04 and 18.04. Uses some of the newer python features.

You can import the conda environment with

```bash
conda env create -f environment.yml
```

(Some of the packages in `environment.yml` file may not be strictly required for the code to run. We will clean this up in the future.)


## Demo

* [Causal Tracing on Mamba](demo/causal_tracing.ipynb)
* [ROME on Mamba](demo/rome_mamba.ipynb)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{sensharma2024locating,
    title={Locating and Editing Factual Associations in Mamba}, 
    author={Arnab Sen Sharma and David Atkinson and David Bau},
    year={2024},
    eprint={2404.03646},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```