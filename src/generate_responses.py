import logging
from typing import Optional

import einops
import numpy as np

import torch
import transformers
from transformers import AutoTokenizer

from src.config import Config
from src.utils import data_utils

logger = logging.getLogger(__name__)

def generate_responses(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates responses for the given dataset.

    Parameters
    ----------
    config : Config
        Typed configuration object from config.py.
    Returns
    -------
    original_questions_array, perturbed_questions_array, responses_array, answers_array : Tuple[np.ndarray, ...]
        Arrays holding the questions, perturbed questions, model responses, and
        the correct answers, all with shapes depending on n_test, n_perturb, and
        n_sample in config.
    """
    logger.info("Loading dataset...")
    dataset = data_utils.load_dataset(config)

    # If needed, load a dictionary of pre-generated perturbed questions.
    perturbed_questions_dict = None
    if config.do_perturb:
        logger.info("Loading perturbed questions dictionary...")
        perturbed_questions_dict = data_utils.load_questions(config)

    # Prepare lists to accumulate results.
    original_questions = [[] for _ in range(len(dataset))]  # shape: (n_test)
    perturbed_questions = [[] for _ in range(len(dataset))] # shape: (n_test)
    responses = [[] for _ in range(len(dataset))]           # shape: (n_test)
    answers = [[] for _ in range(len(dataset))]             # shape: (n_test)

    # Prepare LLaMa model if needed
    if config.model[:2] == 'gpt':
        pipeline = ""
    elif config.model[:5] == 'llama':
        llama_version = config.model.split('-')[1]
        params = config.model.split('-')[2]
        base_or_chat = "" if config.model.split('-')[3] == 'base' else "-chat"
        model = "meta-llama/Llama-" + llama_version + "-" + params + base_or_chat

        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    logger.info("=" * 50)
    logger.info("Generating Responses".center(50))
    logger.info("=" * 50)

    for idx_test, data_entry in enumerate(dataset):
        # Extract the question and answer from the data
        question_text, correct_answer = _extract_question_and_answer(data_entry, config)
        logger.debug("Processing question %d: '%s'", idx_test + 1, question_text)

        # Repeat question for each perturbation & sample
        repeated_qs = einops.repeat(
            np.array([question_text]),
            "b -> b n_perturb n_sample",
            b=1, n_perturb=config.n_perturb, n_sample=config.n_sample
        ).tolist()[0]
        original_questions[idx_test] = repeated_qs

        # Prepare for each test to have shape (n_perturb, n_sample) for responses
        responses[idx_test] = [[] for _ in range(config.n_perturb)]
        answers[idx_test] = [[] for _ in range(config.n_perturb)]

        # Generate or sample perturbed questions
        example_questions, example_responses, example_answers = _get_perturbation_responses(
            idx_test,
            question_text,
            correct_answer,
            perturbed_questions_dict,
            config,
            pipeline, 
            tokenizer
        )

        # example_questions might be shape (n_perturb, 1)
        perturbed_questions[idx_test] = example_questions

        # example_responses is now shape (n_perturb, n_sample)
        responses[idx_test] = example_responses
        answers[idx_test] = example_answers
    
    # Convert lists to np.ndarray for final output
    original_questions_array = np.array(original_questions, dtype=str)
    perturbed_questions_array = np.array(perturbed_questions, dtype=str)
    responses_array = np.array(responses, dtype=str)
    answers_array = np.array(answers, dtype=str)

    logger.info("Response Generation Complete")
    logger.info("=" * 50)

    return original_questions_array, perturbed_questions_array, responses_array, answers_array

def _extract_question_and_answer(data_entry: dict, config: Config) -> tuple[str, str]:
    """
    Extract the question text and answer from a dataset entry, depending on the dataset.
    Raises NotImplementedError for unsupported datasets.
    """
    if config.dataset == "trivia_qa":
        return data_entry["input"], data_entry["answer"]["value"]
    elif config.dataset == "nq":
        return data_entry["input"], data_entry["answer"]
    else:
        raise NotImplementedError(f"Dataset '{config.dataset}' not supported yet.")

def _get_perturbation_responses(
    idx_test: int,
    question_text: str,
    correct_answer: str,
    perturbed_dict: Optional[dict],
    config: Config,
    pipeline: str, 
    tokenizer
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """
    For each perturbation, generate or retrieve a perturbed question,
    then call the model to get responses.

    Returns
    -------
    example_questions, example_responses, example_answers : Three parallel lists
        - example_questions is a list of lists, where each sublist holds the perturbed question(s)
        - example_responses is a flat list containing the generated responses
        - example_answers is a flat list containing the correct answers repeated
          for each generated response
    """
    if config.do_perturb and perturbed_dict:
        # Sample n_perturb perturbed questions from perturbed_dict
        n_perturb = config.n_perturb
        sample_perturb_questions = data_utils.sample_from_perturbed_questions(
            idx_test,
            perturbed_dict,
            config
        )
    else:
        # If not perturbing, use the original question
        n_perturb = 1
        sample_perturb_questions = [question_text]

    example_questions = []
    example_responses = []
    example_answers = []

    # For each perturbation, generate a n_sample responses
    for idx_perturb in range(n_perturb):
        q_perturbed = sample_perturb_questions[idx_perturb]

        # Build the prompt for the model
        if config.prompt_type == 'few_shot':
            prompt = data_utils.sample_to_prompt(q_perturbed, config)
        elif config.prompt_type == 'zero_shot':
            prompt = data_utils.sample_to_prompt_zero_shot(q_perturbed, config)
        else:
            raise NotImplementedError

        # Generate responses from LLM
        model_responses = data_utils.generate_response(prompt, config, pipeline, tokenizer)

        example_questions.append([q_perturbed])
        example_responses.append(model_responses)
        example_answers.append([correct_answer] * len(model_responses))

        logger.debug(
            "Perturbation %d for question '%s': %d responses generated",
            idx_perturb + 1, q_perturbed, len(model_responses)
        )

    return example_questions, example_responses, example_answers
