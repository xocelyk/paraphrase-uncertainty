import json
import logging
import os

from src.config import Config
from src.utils import data_utils

logger = logging.getLogger(__name__)

def generate_questions(config: Config) -> dict:
    """
    Generate and optionally perturb questions from a given dataset
    ('trivia_qa' or 'nq'), storing them in a JSON file.

    Parameters
    ----------
    config : Config
        Typed configuration object from config.py.

    Returns
    -------
    perturbed_questions_dict : dict
        A dictionary mapping each question index to a list of question variants
        (the original plus one or more perturbations).
    """
    logger.info("Loading dataset for: %s", config.dataset)
    dataset = data_utils.load_dataset(config)
    perturbed_questions_dict = {}

    for idx_test, data_entry in enumerate(dataset):
        idx, inp, ans = _extract_data(data_entry, config)
        logger.debug("Processing index=%d, question='%s'", idx_test, inp)

        # Generate perturbed questions for this item
        sample_perturbed_questions = _generate_perturbations(inp, config)

        # Store results
        perturbed_questions_dict[idx] = sample_perturbed_questions

        logger.info("Processed sample index=%d => %d questions total",
                    idx_test, len(sample_perturbed_questions))

    # Save to JSON in an atomic way
    questions_dir = os.path.join(
        os.path.dirname(__file__),
        "questions"
    )
    os.makedirs(questions_dir, exist_ok=True)

    output_path = os.path.join(questions_dir, f"{config.dataset}.json")
    temp_path = output_path + ".tmp"
    logger.info("Saving perturbed questions to %s (writing via %s)", output_path, temp_path)

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(perturbed_questions_dict, f, indent=2, ensure_ascii=False)

        os.replace(temp_path, output_path)
        logger.info("Successfully saved JSON to %s", output_path)
    except OSError as e:
        logger.error("Failed to write JSON file: %s", e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return perturbed_questions_dict

def _extract_data(data_entry, args):
    """
    Extract the index, input, and answer from a single dataset entry,
    depending on the dataset type ('trivia_qa' or 'nq').

    Raises
    ------
    NotImplementedError
        If the dataset is not recognized (i.e., not 'trivia_qa' or 'nq').
    """
    if args.dataset == 'trivia_qa':
        index = data_entry['index']
        inp = data_entry['input']
        ans = data_entry['answer']['value']
    elif args.dataset == 'nq':
        index = data_entry['index']
        inp = data_entry['input']
        ans = data_entry['answer']
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not recognized (expected 'trivia_qa' or 'nq').")

    return index, inp, ans

def _generate_perturbations(question_text, args):
    """
    Create a list of question variants, including the original question,
    plus 'n_perturb' new rephrasings.

    Parameters
    ----------
    question_text : str
        The original question text.
    args : Namespace
        Must include 'n_perturb' among other relevant fields.

    Returns
    -------
    sample_perturbed_questions : list of str
        A list of perturbed question strings, starting with the original question.
    """
    sample_perturbed_questions = [question_text]

    # Generate 'n_perturb' distinct rephrasings
    for i in range(args.n_perturb):
        perturbed_q, sample_perturbed_questions = _attempt_perturbation_loop(
            question_text, sample_perturbed_questions, args
        )
        logger.debug("Perturbation #%d => '%s'", i+1, perturbed_q)

    return sample_perturbed_questions

def _attempt_perturbation_loop(orig_question, question_list, args, max_tries=6):
    """
    Attempt generating a new perturbed question up to 'max_tries' times,
    ensuring we don't produce duplicates. If we exhaust tries, we update
    the base question to the last successful perturbation.

    Returns
    -------
    new_perturbed_q : str
        The newly generated perturbation (not a duplicate of existing ones).
    question_list : list of str
        Updated list with the new question appended.
    """
    num_tries = 0
    new_perturbed_q = None
    while new_perturbed_q in question_list or new_perturbed_q is None:
        new_perturbed_q = data_utils.rephrase_question(orig_question, args)
        num_tries += 1
        if num_tries >= max_tries:
            if len(question_list) > 1:
                # Switch base input to last item in list
                orig_question = question_list[-1]
                logger.debug("Switching base question to '%s'", orig_question)
            num_tries = 0
    question_list.append(new_perturbed_q)
    return new_perturbed_q, question_list
