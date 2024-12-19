# Project Overview

This codebase allows you to run two main workflows:
1. A question generation experiment to produce new or perturbed questions.  
2. A calibration experiment that evaluates model confidence, accuracy, and calibration metrics.

You can optionally generate your own perturbed questions using the “generate” workflow. However, this repository already includes prepopulated question files in the "questions/" directory, so feel free to skip the generation step if you just want to run the calibration with existing questions.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Generate Questions](#generate-questions)
  - [Run Calibration](#run-calibration)
- [Project Structure](#project-structure)

---

## Installation

1. Clone this repository.
2. Install required Python packages:   ```
   pip install -r requirements.txt   ```
3. Set your OpenAI API key in an environment variable:   ```
   export OPENAI_API_KEY="sk-..."   ```

---

## Configuration

All runtime settings are in the file `config.yaml`. Below is a short example:

```yaml
dataset: "trivia_qa"
split: "validation"
model: "gpt-3.5-turbo"
n_test: 100
do_perturb: true
n_perturb: 6
n_sample: 2
temperature: 1.0
max_tokens: 100
stop: null
prompt_type: "few_shot"
preload_responses: false
preload_accuracy: false
preload_uncertainty: false
preload_calibration: false
save_results: true
verbose: true
```


Key fields include:
- **dataset**: Which dataset to load (e.g., "nq" or "trivia_qa").  
- **model**: The OpenAI model or local model to use (e.g., "gpt-3.5-turbo").  
- **n_test**: Number of data points to process from the dataset.  
- **do_perturb** / **n_perturb**: Whether to generate additional perturbations per question.  
- **save_results**: Whether to save outputs (responses, metrics etc.) to disk.  
- **preload_** flags (e.g., preload_responses): Whether to load previously generated data from disk.  
- **device**: Auto-chosen or specified (CPU, CUDA, MPS).  

---

## Usage

All commands are run from the project’s root directory.

### Generate Questions
Use the "generate" subcommand to produce new or perturbed questions based on the loaded dataset:

```
python -m src.main generate
```

This workflow:
1. Loads the specified dataset (e.g., "trivia_qa").  
2. Applies optional text perturbations to generate rephrased versions of each question.  
3. Saves the augmented questions to a JSON file in the "questions/" directory.

Although the codebase supports generating perturbed questions dynamically, the "questions/" directory is already prepopulated with ready-to-use question files. Hence, run this step only if you need to generate fresh questions or want more custom perturbations.

### Run Calibration
Use the "calibrate" subcommand to evaluate the model’s performance and calibration:

```
python -m src.main calibrate
```

This will:
1. Load or generate responses to questions (optionally using the perturbed ones you created).  
2. Evaluate uncertainty (via embeddings or other methods).  
3. Compute accuracy against known answers.  
4. Measure calibration metrics (i.e., how confidently the model makes correct vs. incorrect predictions).  
5. Save results in a subfolder of the `results/` directory.

---

## Project Structure

Below is an overview:
- **main.py**  
  Entry point that can run “generate” (question generation) or “calibrate” (calibration experiment).
- **config.yaml**  
  Main configuration file (dataset, model, device, etc.).
- **src/**  
  - **config.py**: Loads/parses `config.yaml` and environment settings.  
  - **generate_questions.py**: Logic for generating or perturbing questions.  
  - **generate_responses.py**: Retrieves answers from the model for each question.  
  - **evaluate_accuracy.py**, **evaluate_uncertainty.py**, **evaluate_calibration.py**: Functions for computing accuracy, uncertainty, and calibration metrics.  
  - **utils/data_utils.py**: Helpers for loading and manipulating data.
- **requirements.txt**  
  Dependencies (OpenAI, scikit-learn, transformers, etc.).  
- **questions/**  
  JSON files containing the questions for different datasets, both original and perturbed.  
- **results/**  
  Output data (responses, pickled evaluations, calibration metrics).