# AIAP11 Assessment


## Table of Contents
 - [Candidate Information](#a-candidate-information)
 - [Folder structure](#b-folder-structure)
 - [Pipeline and Configuration](#c-instructions-for-pipeline-and-configuration)
 - [Pipeline Flow Information](#d-pipeline-flow-information)
 - [Summary of EDA](#e-summary-of-eda-findings)
 - [Model Justification](#f-justification-of-model-choice)
 - [Evaluation of Metrics](#g-evaluation-of-model-metrics)
 - [Other Considerations](#h-other-considerations)

---

## A. Candidate Information

**Name:** Low Guangwen Daniel

**Email:** dlow017@e.ntu.edu.sg

---

## B. Folder Structure

Overview of the submitted folder and the folder structure

```bash
tree -I __pycache__
```

---

## C. Instructions for Pipeline and Configuration

### C.1 Deployment Information

The entire file system was created in a virtual environment using conda.

Please run the following steps in bash to replicate the virtual environment:

```bash
conda create -n aiap11 python=3.9.0
conda activate aiap11
conda install --file requirements.txt
```

Please run the following steps in bash to run the run.sh file

```bash
./run.sh
```


### C.2 Pipeline Configuration

The model config files can be found in config.yaml within the config folder of the src. The configurable parameters are categrized into:

 1. Data Location and table names
 2. Preprocess parameters
 3. Model parameters

```bash
src
|-- config
    |-- config.yaml
    `-- config_load.py
```


## D. Pipeline Flow Information

Description of logical steps/flow of the pipeline. If you find it useful, please feel free to include suitable visualization aids (eg, flow charts) within the README.

```mermaid


```

---

## E. Summary of EDA Findings

Overview of key findings from the EDA conducted in Task 1 and the choices made in the pipeline based on these findings, particularly any feature engineering. Please keep the
details of the EDA in the `.ipynb`, this section should be a quick summary.

Hence, decided to keep the following columns 
["branch", "country", "first_time", "with_child", "SGD_price"]

---

## F. Justification of Model Choice


### F.1 Logistic Regression

Baseline Classifier Model

### F.2 Decision Tree Classifier

The decision tree was chosen as it would allow us to form a baseline metrics on how well basic trees can perform predictions on the dataset

### F.3 Random Forest Classifier

- Random Forest Model chosen over other decision tree models for example XGBoost due to the commercial nature of the question (hotel bookings)
- Typically in commmercial situations, changes in data can rapid and hence needed a model that can deal with variance errors better.
- Boosting models for eg XGBoost employs iterative strategy for adjusting an observation's weight based on the previous wrongly classified information, hence boosting models generally result in better prediction outcomes but may not always generalize well.
- Bagging models like Random Forest Classifiers create extra data by bagging (sampling with replacement) and create multiple parallel models of which mean predictions are chosen. Bagging models hence deal better with data randomness and data variation.


---

## G. Evaluation of Model Metrics

Evaluation of the models developed. Any metrics used in the evaluation should also be
explained.

---

## H. Other Considerations

1. Ethical Considerations
    - Unsure whether to use country as a column even though it is somewhat predictive
    - Even if column is highly predictive, little actionable policies can be formulated
    - Negative impact of leaked algorithm selection might hurt the business.

---

## Evaluation

1. Appropriate data preprocessing and feature engineering
2. Appropriate use and optimisation of algorithms/models
3. Appropriate explanation for the choice of algorithms/models
4. Appropriate use of evaluation metrics
5. Appropriate explanation for the choice of evaluation metrics
6. Understanding of the different components in the machine learning pipeline

### Penalties
1. Incorrect format for `requirements.txt`
2. `run.sh` fails upon execution
3. Poorly structured `README.md`
4. Disorganised code that fails to make use of functions and/or classes for reusability
5. Machine learning pipeline built using Jupyter Notebooks.