# Supplementary Code: Multi-Fidelity Batch Bayesian Optimization and Bioprocess Simulation

This repository contains the supplementary code accompanying the paper:

📄 **“Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization”**  
*Authors: Adrian Martens, Mathias Neufang, Alessandro Butté, Moritz von Stosch, Antonio del Rio Chanona, Laura Marie Helleckes*  
[arXiv:2508.10970](https://arxiv.org/abs/2508.10970)  

---

## Overview

This repository provides the implementation of the methods and experiments described in the paper.
It is intended to help readers reproduce key results, explore the methodology, and extend it for related research.

The code is written in **Python 3.11** and includes:

* Bioprocess simulation model
* Proposed BO workflow
* Parallelized optimization test runs
* Experimental results
* Visualization utilities for reproducing the figures in the paper

---

## Repository Structure

```
.
├── results/                            # Results presented in the paper
├── notebooks/                          # Jupyter notebooks for visualization and demonstration
│   ├── cho_model_showcase.ipynb        # Simulation showcase of CHO model
│   ├── graphs.ipynb                    # Script to generate figures for the paper
│   └── performance_plots.ipynb         # Analyze results and create performance plots
├── run/                                # Scripts to run experiments in parallel
│   ├── run_all_bo_scenarios.py         # Run all BO test scenarios in parallel
│   ├── run_industrial_scenarios.py     # Run industrial scenario experiments
│   └── run_single_bo_scenario.py       # Helper script to run a single BO scenario
├── src/mfbo4bio/                       # Core implementation of MF-BO framework
│   ├── clone_creator.py                # Create varied clone parameter sets
│   ├── conditions_data.py              # Store and manage clone parameters
│   ├── industrial_methods.py           # Implement industrial baseline approaches
│   ├── mfbo_GIBBON.py                  # MF-BO with GIBBON acquisition function
│   ├── mfbo_qLogEI.py                  # MF-BO with qLogEI acquisition function
│   ├── mfbo_qUCB.py                    # MF-BO with qUCB acquisition function
│   ├── optimization.py                 # Custom grid search optimization routines
│   ├── utils.py                        # General helper functions
│   └── virtual_lab.py                  # Bioprocess simulation code
├── .gitignore                          # Git ignore rules
├── .pre-commit-config.yaml             # Pre-commit hook configuration
├── LICENSE                             # MIT license
├── pyproject.toml                      # Project/package configuration
├── README.md                           # Project description and instructions
├── requirements.txt                    # Python dependencies
└── uv.lock                             # Lock file for uv package manager

```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/adrian-martens/mf-bo4bio.git
   cd mf-bo4vio
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   pip install -e .
   ```

---

## Usage

### 1. Running Experiments

To reproduce the experiments from the paper, use the scripts in the `run/` directory.  
All runs are executed in parallel.  

Run **all Bayesian Optimization (BO) scenarios**:

```bash
python run/run_all_bo_scenarios.py      # for the BO workflow
python run/run_industrial_scenarios.py  # for the insustrial comparison scenario
```

### 2. Reproducing Figures

Notebooks for visualization and figure creation are provided in `notebooks/`.
For example, to recreate the figures from the paper:

```bash
jupyter notebook notebooks/graphs.ipynb
```

Other useful notebooks:
* `cho_model_showcase.ipynb` – showcase of the CHO model simulation
* `performance_plots.ipynb` – performance analysis and plotting

---

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{martens2025bioprocessdevelopmentscales, 
  title         = {Bioprocess Development Across Scales Using Multi-Fidelity Batch Bayesian Optimization}, 
  author        = {Adrian Martens and Mathias Neufang and Alessandro Butté and Moritz von Stosch and Antonio del Rio Chanona and Laura Marie Helleckes}, 
  year          = {2025}, 
  eprint        = {2508.10970}, 
  primaryClass  = {q-bio.QM}, 
  url           = {https://arxiv.org/abs/2508.10970}, 
  }
```

---

## License

This code is released under the **MIT License** (see `LICENSE` file).
