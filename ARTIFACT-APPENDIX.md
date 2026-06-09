# Artifact Appendix

## Paper

**Title:** CoP: Coordinated Perturbation for Controlled Disclosure Under Local Differential Privacy

**Authors:** Sandaru Jayawardana, Ming Ding and Kanchana Thilakarathna

**Venue:** PETS / PoPETs 2026

**License:** Apache License 2.0

## 1. Overview

This artifact provides the implementation of the **CoP perturbation pipeline** proposed in the paper.

The purpose of this artifact is to allow reviewers to inspect and execute the main perturbation mechanism. The artifact focuses on:

1. constructing CoP perturbation mechanisms from empirical dependency information;
2. generating locally perturbed outputs under a given privacy budget;
3. running a small dummy-data example to check functionality;
4. documenting the required inputs and expected outputs of the perturbation pipeline.

This artifact is not intended to reproduce every figure or full-scale experiment in the paper. Instead, it supports the core implementation of the CoP mechanism and demonstrates how the perturbation pipeline operates.

## 2. Artifact Scope

The artifact includes:

* the CoP mechanism implementation;
* the optimized randomized-response mechanism used inside CoP;
* the convex optimization routine used to construct optimized local mechanisms;
* a perturbation notebook or script demonstrating the pipeline;
* a small dummy example for reviewers to run quickly.

The artifact does not include:

* full paper-scale experimental reproduction;
* all baseline mechanisms;
* all plotting scripts;
* raw third-party datasets, if redistribution is restricted by their original licenses.

## 3. Repository Structure

The repository is organized as follows.

```text
.
├── ARTIFACT-APPENDIX.md
├── README.md
├── LICENSE
├── requirements.txt
├── mechanisms/
│   ├── privacy_mechanism.py
│   └── CoP/
│       ├── cop.py
│       ├── cop_multi_thread.py
│       ├── optimized_rr.py
│       └── convex_optimizer.py
├── notebooks/
│   └── cop_perturb.ipynb
├── scripts/
│   ├── run_cop_smoke_test.py
│   └── run_cop_perturbation.py
├── datasets/
│   └── dummy/
└── results/
    └── dummy/
```

## 4. Main Components

### 4.1 CoP mechanism

The main CoP implementation is in:

```text
mechanisms/CoP/cop.py
```

This file implements the CoP perturbation mechanism. The mechanism takes as input:

* an ordered list of attributes;
* the alphabet of each attribute;
* conditional mass functions between attributes;
* dependency scores such as PMI and MI;
* a privacy budget;
* threshold parameters controlling which dependencies are used.

The mechanism first constructs optimized perturbation mechanisms for selected dependent attribute pairs. It then perturbs each input record by randomly selecting an unvisited attribute and traversing the selected dependency structure to generate coordinated perturbed outputs.

### 4.2 Multithreaded CoP mechanism

The optional multithreaded version is in:

```text
mechanisms/CoP/cop_multi_thread.py
```

This version constructs CoP mechanisms in parallel across attributes. It is useful for larger attribute sets but is not required for the smoke test.

### 4.3 Optimized randomized response

The optimized randomized-response mechanism is in:

```text
mechanisms/CoP/optimized_rr.py
```

This component constructs a local perturbation matrix for a given privacy budget. It is used internally by CoP when generating attribute-specific perturbation mechanisms.

### 4.4 Convex optimizer

The optimizer is in:

```text
mechanisms/CoP/convex_optimizer.py
```

This component uses CVXPY to solve the optimization problem used to construct the optimized local perturbation matrix.

### 4.5 Perturbation notebook

The perturbation notebook is in:

```text
notebooks/cop_perturb.ipynb
```

This notebook demonstrates the CoP perturbation pipeline interactively.

## 5. Hardware Requirements

The artifact does not require a GPU.

Recommended hardware:

```text
CPU: 2 or more cores
RAM: 4 GB or more
Disk: 1 GB free space
GPU: Not required
```

For larger datasets or larger attribute alphabets, more memory and CPU cores may reduce runtime.

## 6. Software Requirements

The artifact was developed for Python 3.

Required packages include:

```text
numpy
pandas
scipy
cvxpy
highspy
matplotlib
scikit-learn
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

The artifact uses CVXPY with the HiGHS solver for the optimization step.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/SandaruJayawardana/CoP-artifacts-PETS
cd CoP-artifacts-PETS
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Input Format

The CoP perturbation pipeline requires the following inputs.

### 8.1 Attribute list

A list specifying the order of attributes:

```python
ordered_attribute_list = ["A", "B", "C"]
```

### 8.2 Attribute alphabets

A dictionary specifying the possible values of each attribute:

```python
alphabet_dict = {
    "A": ["0", "1"],
    "B": ["0", "1"],
    "C": ["0", "1"]
}
```

### 8.3 Conditional mass functions

A dictionary containing conditional distributions for attribute pairs:

```python
CMF_dict = {
    "A B": np.array([[0.8, 0.2], [0.3, 0.7]]),
    "A C": np.array([[0.6, 0.4], [0.4, 0.6]]),
    "B A": np.array([[0.7, 0.3], [0.2, 0.8]]),
    "B C": np.array([[0.5, 0.5], [0.4, 0.6]]),
    "C A": np.array([[0.6, 0.4], [0.5, 0.5]]),
    "C B": np.array([[0.7, 0.3], [0.3, 0.7]])
}
```

Each key has the form:

```text
"source_attribute target_attribute"
```

### 8.4 Dependency scores

The mechanism also uses dependency scores such as PMI and MI:

```python
pmi_dict = {
    "A B": np.array([[0.9, 0.1], [0.2, 0.8]]),
    "A C": np.array([[0.7, 0.3], [0.3, 0.7]]),
    "B A": np.array([[0.8, 0.2], [0.2, 0.8]]),
    "B C": np.array([[0.5, 0.5], [0.4, 0.6]]),
    "C A": np.array([[0.6, 0.4], [0.5, 0.5]]),
    "C B": np.array([[0.7, 0.3], [0.3, 0.7]])
}

mi_dict = {
    "A B": 0.5,
    "A C": 0.4,
    "B A": 0.5,
    "B C": 0.3,
    "C A": 0.4,
    "C B": 0.3
}
```

These values are used to decide which dependent attributes should be included in the coordinated perturbation process.

## 9. Running the Smoke Test

The smoke test checks that the CoP mechanism can be constructed and used to perturb a small dummy dataset.

Run $test.ipynb$.

The smoke test performs the following steps:

1. loads or creates a small dummy dataset;
2. defines attribute alphabets;
3. constructs dummy conditional mass functions;
4. constructs dependency-score dictionaries;
5. initializes the CoP mechanism;
6. perturbs each record using a chosen privacy budget;
7. writes the perturbed outputs to the `results/` directory.
8. Evaluate MSE.
9. Plot MSE vs Privacy budget.
   
Expected output:

Graph of MSE vs Privacy budget.

## 10. Running the Perturbation Pipeline

## 11. License

The source code is released under the Apache License 2.0.

Datasets are subject to their original licenses. If raw datasets cannot be redistributed, the repository provides instructions for preparing compatible input files.

## 12. Contact

For questions about this artifact, contact:

```text
Sandaru Jayawardana (sjay9734@uni.sydney.edu.au)
```
