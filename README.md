# Artifact Appendix

## Paper

**Title:** CoP: Coordinated Perturbation for Controlled Disclosure Under Local Differential Privacy

**Authors:** Sandaru Jayawardana, Ming Ding and Kanchana Thilakarathna

**Venue:** PETS / PoPETs 2026

**License:** Apache License 2.0

## 1. Overview

This artifact provides the implementation of the **CoP perturbation pipeline** proposed in the paper.

The artifact focuses on:

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
├── ARTIFACT-APPENDIX.md
├── README.md
├── LICENSE
├── requirements.txt
├── mechanisms/
│   ├── privacy_mechanism.py
│   └── CoP/
│       ├── cop.py
│       ├── cop_multithread.py
│       ├── optimized_rr.py
│       └── convex_optimizer.py
├── notebooks/
│   ├── cop_dummy_pipeline.ipynb
├── datasets/
│   └── dummy.csv
├── results/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── cmf_pmf_cal.py
    ├── data_perturb.py
    ├── eval_perturbed.py
    ├── mi_compute.py
    ├── mutual_information.py
    ├── normalize_error_matrix.py
    ├── pcc_compute.py
    ├── per_attribute_privacy_budget_compute.py
    ├── pmi_cal.py
    ├── theoretical_privacy_leakage.py
    └── util_functions.py
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
jupyter 
nbconvert 
ipykernel
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

The artifact uses CVXPY with the HiGHS solver for the optimization step.

## 7. Installation

The artifact can be installed either using a local Python virtual environment or using Docker.

### 7.1 Option A: Local Installation

Clone the repository:

```bash
git clone https://github.com/SandaruJayawardana/CoP-artifacts-PETS
cd CoP-artifacts-PETS
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 7.2 Option B: Docker Installation

We also provide a Dockerfile to support a reproducible build environment for artifact evaluation. The Docker image installs the required Python dependencies, notebook execution tools, and HiGHS solver support.

Build the Docker image from the repository root:

```bash
docker build -t cop-artifact .
```

Run the container interactively:

```bash
docker run --rm -it cop-artifact
```

Alternatively, to open the example notebook in a browser using Jupyter, run:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD:/artifact" \
  -w /artifact \
  cop-artifact \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/artifact
```

Then open the Jupyter URL printed in the terminal and navigate to:

```text
notebooks/cop_dummy_pipeline.ipynb
```

The volume mount `-v "$PWD:/artifact"` connects the local repository folder to `/artifact` inside Docker. Therefore, any results written to `/artifact/results/` inside Docker will also appear in the local `results/` folder.

## 8. Input Format

The CoP perturbation pipeline requires a multidimensional categorical dataset. Each row corresponds to one user record, and each column corresponds to one attribute.

For the provided smoke test, the input dataset is:

```text
datasets/dummy.csv
```

The notebook automatically loads the dataset, extracts the attribute list, estimates dependency information, initializes the CoP mechanism, and evaluates the perturbed outputs.

### 8.1 Attribute List

The attribute list specifies the order of attributes used by the mechanism. In the smoke-test notebook, this list is obtained directly from the input dataset:

```python
COLUMNS = data.columns.to_list()
```

For example, if the dataset contains three attributes, the attribute list may be:

```python
ordered_attribute_list = ["A", "B", "C"]
```

The order of attributes is important because input records and perturbed output records follow this same order.

### 8.2 Attribute Alphabets

The alphabet of an attribute is the set of possible values that the attribute can take. The smoke-test notebook estimates the attribute alphabets from the dataset:

```python
CMF_dict, alphabet_dict = get_cmf_pmf_dict_with_alphabet(
    data=data,
    is_pmf=False
)
```

Here, `alphabet_dict` maps each attribute to its possible values.

### 8.3 Dependency Information

CoP uses prior dependency information to coordinate perturbation across attributes. In the smoke-test notebook, this information is estimated from the input data:

```python
mi_dict = get_mi_dict(data=data, COLUMNS=COLUMNS)
PMI_dict = get_pmi_dict(data=data)
```

The conditional probability matrices are stored in:

```python
CMF_dict
```

These dependency statistics are then passed to the CoP mechanism:

```python
cop_mechanism = CoP_Mechanism(
    ordered_attribute_list=COLUMNS,
    alphabet_dict=alphabet_dict,
    CMF_dict=CMF_dict,
    pmi_dict=PMI_dict,
    mi_dict=mi_dict
)
```

## 9. Running the Smoke Test

The smoke test checks that the CoP mechanism can be constructed and used to perturb a small dummy dataset. It also validates the generated perturbed datasets and evaluates their utility.

The smoke-test notebook is located at:

```text
notebooks/cop_dummy_pipeline.ipynb
```

### 9.1 Run Locally

After completing the local installation, run:

```bash
jupyter notebook notebooks/cop_dummy_pipeline.ipynb
```

Then execute all cells in the notebook.

Alternatively, the notebook can be executed from the command line:

```bash
jupyter nbconvert \
  --to notebook \
  --execute notebooks/cop_dummy_pipeline.ipynb \
  --output executed_cop_dummy_pipeline.ipynb \
  --output-dir results
```

### 9.2 Run with Docker

After building the Docker image, execute the notebook inside Docker using:

```bash
docker run --rm -it \
  -v "$PWD:/artifact" \
  -w /artifact \
  cop-artifact \
  jupyter nbconvert \
  --to notebook \
  --execute notebooks/cop_dummy_pipeline.ipynb \
  --output executed_cop_dummy_pipeline.ipynb \
  --output-dir results
```

This command writes the executed notebook and generated outputs to the local `results/` directory.

### 9.3 Smoke-Test Steps

The smoke test performs the following steps:

1. loads the dummy dataset from `datasets/dummy.csv`;
2. removes missing values, if any;
3. extracts the ordered attribute list;
4. estimates attribute alphabets and conditional probability matrices;
5. computes mutual information and pointwise mutual information dictionaries;
6. initializes the CoP mechanism;
7. perturbs the dataset under multiple privacy budgets;
8. saves the perturbed outputs to the `results/` directory;
9. validates that the perturbed datasets have the expected format;
10. evaluates the perturbed datasets against the original data.

### 9.4 Expected Outputs

After successful execution, the `results/` directory should contain perturbed datasets such as:

```text
dummy_perturbed_eps_1.csv.zip
dummy_perturbed_eps_2.csv.zip
dummy_perturbed_eps_4.csv.zip
dummy_perturbed_eps_10.csv.zip
dummy_perturbed_eps_15.csv.zip
```

If the notebook is executed using `nbconvert`, the following file is also created:

```text
executed_cop_dummy_pipeline.ipynb
```

The notebook also reports the evaluation results generated by `Evaluate_Perturbed_Data`. Depending on the evaluation configuration, this may include utility-error values such as MSE and corresponding plots over different privacy budgets.

Expected output:

```text
Perturbed datasets are generated, validated, and evaluated successfully.
```

The expected plot is:

```text
MSE vs Privacy Budget
```

## 10. License

The source code is released under the Apache License 2.0.

Datasets are subject to their original licenses. If raw datasets cannot be redistributed, the repository provides instructions for preparing compatible input files.

## 11. Contact

For questions about this artifact, contact:

```text
Sandaru Jayawardana (sjay9734@uni.sydney.edu.au)
```
