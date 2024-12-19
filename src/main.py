import argparse
import logging
import os
import pickle

import openai

from src.config import config
from src.evaluate_accuracy import evaluate_accuracy
from src.evaluate_calibration import evaluate_calibration
from src.evaluate_uncertainty import evaluate_uncertainty
from src.generate_questions import generate_questions
from src.generate_responses import generate_responses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenAI with key from config
openai.api_key = config.openai_api_key

def calibrate_experiment():
    """
    Runs the calibration flow using fields from 'config'.
    """
    # Prepare directory under data/results (one level above src/)
    results_dir = os.path.join(
        "results",
        f"{config.dataset}_{config.do_perturb}_{config.n_test}_"
        f"{config.n_perturb}_{config.n_sample}_{config.model}"
    )
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Generating responses with model=%s, device=%s", config.model, config.device)
    
    if config.preload_responses:
        with open(os.path.join(results_dir, "original_questions.pkl"), "rb") as f:
            original_questions = pickle.load(f)
        with open(os.path.join(results_dir, "perturbed_questions.pkl"), "rb") as f:
            perturbed_questions = pickle.load(f)
        with open(os.path.join(results_dir, "responses.pkl"), "rb") as f:
            responses = pickle.load(f)
        with open(os.path.join(results_dir, "answers.pkl"), "rb") as f:
            answers = pickle.load(f)
    else:
        original_questions, perturbed_questions, responses, answers = generate_responses(config)
    
    if config.save_results:
        _save_pickle(results_dir, "original_questions.pkl", original_questions)
        _save_pickle(results_dir, "perturbed_questions.pkl", perturbed_questions)
        _save_pickle(results_dir, "responses.pkl", responses)
        _save_pickle(results_dir, "answers.pkl", answers)

    if config.preload_uncertainty:
        with open(os.path.join(results_dir, "uncertainty.pkl"), "rb") as f:
            uncertainty_results = pickle.load(f)
    else:
        uncertainty_results = evaluate_uncertainty(responses, config, original_questions=original_questions)

    if config.save_results:
        _save_pickle(results_dir, "uncertainty.pkl", uncertainty_results)

    if config.preload_accuracy:
        with open(os.path.join(results_dir, "accuracy.pkl"), "rb") as f:
            accuracy_results = pickle.load(f)
    else:
        accuracy_results = evaluate_accuracy(original_questions, responses, answers, config)
        
    if config.save_results:
        _save_pickle(results_dir, "accuracy.pkl", accuracy_results)

    if config.preload_calibration:
        with open(os.path.join(results_dir, "calibration.pkl"), "rb") as f:
            calibration_results = pickle.load(f)
    else:
        calibration_results = evaluate_calibration(uncertainty_results, accuracy_results)
    
    if config.save_results:
        _save_pickle(results_dir, "calibration.pkl", calibration_results)
        
def questions_experiment():
    """
    Runs the question generation workflow.
    """
    logger.info("Generating questions for dataset=%s", config.dataset)
    generate_questions(config)

def _save_pickle(directory: str, file_name: str, data):
    path = os.path.join(directory, file_name)
    with open(path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibrate or generate experiments.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    subparsers.add_parser("calibrate", help="Run the calibration experiment.")
    subparsers.add_parser("generate", help="Run the question generation.")

    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate_experiment()
    elif args.command == "generate":
        questions_experiment()
    else:
        parser.print_help()
    
    