# Project Overview

This repository contains the code to run the calibration experiments for the paper "Mapping from Meaning: Addressing the Miscalibration of Prompt-Sensitive Language Models".

The repository allows you to run two main workflows:

1. Paraphrase generation: Generate paraphrases of the original questions in a dataset. Either `trivia_qa` or `nq`.
2. Calibration experiment: Use the generated paraphrases to run a calibration experiment using paraphrase perturbation sampling. Compare the calibration performance over different uncertainty metrics, and different perturbation sampling parameters.

Optionally, we also provide a dataset of 6 paraphrases for 1000 questions in the both `trivia_qa` and `nq`. These are the paraphrases used in the paper. You can run the calibration experiment with these paraphrases, or overwrite them by generating your own.

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
4. Set your Hugging Face access token if you plan to use LLaMa as the LLM model: ```
   huggingface-cli login   ```

---

## Configuration

All runtime settings are in the file `config.yaml`. Below is a short example:

```yaml
dataset: "trivia_qa"
split: "validation"
model: "gpt-3.5-turbo"
n_test: 1000
do_perturb: true
n_perturb: 6
n_sample: 2
temperature: 1.0
max_tokens: 100
eigenvalues_threshold: 0.5
graph_mode: "entailment"
stop: null
prompt_type: "few_shot"
preload_responses: false
preload_accuracy: false
preload_uncertainty: false
preload_calibration: false
save_results: true
verbose: true
```

Field descriptions:
- **dataset**: Which dataset to load (either "trivia_qa" or "nq").  
- **split**: Which split of the dataset to load (either "train", "validation", or "test"). Default is "validation".
- **model**: The LLM model to use for question answering (e.g., "gpt-3.5-turbo", "llama-2-7b-base", "llama-3.3-70b-chat").
- **n_test**: Number of examples to test from the dataset.
- **do_perturb**: Whether to do perturbation sampling for response generation, or to use the original question. If false, **n_perturb** is ignored.
- **n_perturb**: The number of perturbed questions to sample from the perturbed question set, for each original question, without replacement. If using the pre-generated paraphrases, this should be 6 at most.
- **n_sample**: The number of responses to generate for each perturbed question.
- **temperature**: The temperature to use for response generation.
- **max_tokens**: The maximum number of tokens to generate for each response.
- **eigenvalues_threshold**: The maximum eigenvalue to use for determining the eigenvector basis used in the graph-based embedding module in `spectral_uncertainty.py`.
- **graph_mode**: The metric used to determine the weights in the adjacency matrix in the graph-based embedding module. Supports "entailment" and "contradiction".
- **stop**: The stop sequence to use for response generation.
- **prompt_type**: The type of prompt to use for response generation ("few_shot" or "zero_shot").
- **preload_** flags (e.g., preload_responses): Whether to load previously generated data from disk.
- **save_results**: Whether to save outputs (responses, metrics etc.) to disk.
- **verbose**: Whether to print verbose output to the console.

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

Although the codebase supports generating perturbed questions dynamically, the "questions/" directory is already prepopulated with ready-to-use question files. So, run this step only if you need to generate fresh questions or want more custom perturbations.

### Run Calibration
Use the "calibrate" subcommand to evaluate the model’s performance and calibration:

```
python -m src.main calibrate
```

This will:
1. Load or generate responses to questions (optionally using the perturbed ones you created).  
2. Evaluate uncertainty (via embeddings or other methods).  
3. Compute accuracy against known answers. Uses GPT-3.5 for grading.
4. Measure calibration of uncertainty metrics (i.e., how well the uncertainty metrics predict the accuracy of the model).
5. Save results in a subfolder of the `results/` directory.

The following uncertainty metrics are recorded:
- `predictive_entropy`: The entropy of the model's distribution over the responses ("Entropy" in the paper).
- `rouge_l_uncertainty`: A measure of how similar the model's responses are based on the Rouge-L score. "Lexical Similarity" in the paper.
- `semantic_entropy`: The entropy of the model's distribution over the responses, based on semantic clusters of the responses, following [Kuhn et al. (2023)](https://arxiv.org/abs/2302.09664). "Semantic Entropy" in the paper.
- `spectral_clusters`: The `U_{EigV}` metric from [Lin et al. (2024)](https://arxiv.org/abs/2305.19187), called the same in the paper. Extends the idea of calculating the number of distinct semantic sets in the responses to a continuous metric.
- `spectral_variance_sampling_uncertainty`: The part of the total uncertainty attributed to inter-sample variance in the spectral embedding space. Called "Variance (AU)" or "aleatoric uncertainty" in the paper.
- `spectral_variance_perturbation_uncertainty`: The part of the total uncertainty attributed to intra-sample variance in the spectral embedding space. Called "Variance (EU)" or "epistemic uncertainty" in the paper.
- `spectral_variance_total_uncertainty`: The total uncertainty, calculated as the sum of the above metrics. Called "Variance (Total)" or "total uncertainty" in the paper.

---

## Project Structure

Below is an overview of the project structure:
- **main.py**  
  Entry point that can run “generate” (question generation) or “calibrate” (calibration experiment).
- **config.yaml**  
  Main configuration file (dataset, model, device, etc.).
- **src/**  
  - **config.py**: Loads/parses `config.yaml` and environment settings.  
  - **generate_questions.py**: Logic for generating or perturbing questions.  
  - **generate_responses.py**: Retrieves answers from the model for each question.  
  - **evaluate_accuracy.py**, **evaluate_uncertainty.py**, **evaluate_calibration.py**: Functions for computing accuracy, uncertainty, and calibration metrics.
  - **evaluate_uncertainty.py**: Computes the uncertainty metrics.
  - **evaluate_calibration.py**: Computes the calibration of the uncertainty metrics.
- **utils/**:
  - **data_utils.py**: Helpers for loading and manipulating data.
- **models/**:
  - **embedding/**: Contains the NLI model used for graph-based embedding.
  - **inference/**: Contains the GPT API wrapper used for question answering, as well as paraphrase generation and grading.
- **metrics/**:
  - **accuracy/**: Contains the accuracy metrics.
  - **uncertainty/**: Contains the uncertainty metrics.
- **requirements.txt**  
  Dependencies (OpenAI, scikit-learn, transformers, etc.).  
- **questions/**  
  JSON files containing the questions for different datasets, both original and perturbed.  
- **results/**  
  Output data (responses, pickled evaluations, calibration metrics).