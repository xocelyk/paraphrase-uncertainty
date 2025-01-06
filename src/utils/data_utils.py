import json
import os

import datasets
import numpy as np
import openai
import pandas as pd
import torch

import src.models as models
from src.config import Config, config

device = config.device
openai.api_key = config.openai_api_key

def load_questions(config: Config):
    questions_dir = "questions"
    questions_filename = f'{config.dataset}.json'
    questions_filename = os.path.join(questions_dir, questions_filename)
    questions = load_json(questions_filename)
    return questions

def load_json(filename):
    with open(filename, 'r') as f:
        dic = json.load(f)
        dic = {int(k): v for k, v in dic.items()}
    return dic
    
def sample_from_perturbed_questions(idx, perturbed_questions, config: Config):
    # The first question in perturbed_questions is the original question
    n_perturb = config.n_perturb
    perturbed_questions = perturbed_questions[idx]
    assert len(list(set(perturbed_questions))) >= n_perturb, f"Number of perturbed questions ({len(perturbed_questions)}) is less than the number of perturbations requested ({n_perturb})."
    
    # We sample n_perturb questions from perturbed_questions[1:n_perturb + 1] because the first question is the original question
    perturbed_questions = np.random.choice(perturbed_questions[1:], n_perturb, replace=False)
    return perturbed_questions

def sample_to_prompt(question, config: Config, full_sentence_response=False, **kwargs):
    # We use the same prompts as the Meta Llama paper for NQ and TriviaQA https://arxiv.org/pdf/2302.13971
    if isinstance(question, list):
        return [sample_to_prompt(q, **kwargs) for q in question]
    if config.dataset == 'trivia_qa':
        return f"""Answer these questions: 
    Q: In Scotland, a bothy/bothie is a?
    A: House
    Q: {question}
    A:"""

    elif config.dataset == 'nq':
        return f"""Answer these questions:
    Q: Who sang who wants to be a millionaire in high society?
    A: Frank Sinatra
    Q: {question}
    A:"""

    else:
        raise NotImplementedError

def sample_to_prompt_zero_shot(question, **kwargs):
    if isinstance(question, list):
        return [sample_to_prompt_zero_shot(q, **kwargs) for q in question]
    return question

def perturb_sample(sample, config: Config): 
    return rephrase_question(sample, config)
    
def rephrase_question(sample: str, config: Config):
    prompt = '''Rephrase the following question in your own words. The rephrased question should preserve the meaning of the original question, but be worded differently. The answer to the rephrased question should be the same as the answer to the original question. You can be creative.
    Original question: What is the capital city of Australia?
    Rephrased question: Which Australian city serves as the country's capital?
    Original question: {}
    Rephrased question:'''.format(sample)
    rephrase_config = config.copy()
    rephrase_config.n_sample = 1
    rephrase_config.model = 'gpt-4'
    rephrase_config.temperature = 0.9
    response = generate_response(prompt, rephrase_config)[0]
    return response

def load_dataset(config: Config, shuffle=False):
    # Load TriviaQA dataset
    if config.dataset == 'trivia_qa':
        data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=config.split)
        id_mem = set()

        def remove_dups(batch):
            if batch['question_id'][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch['question_id'][0])
            return batch

        data = data.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
        assert pd.Series([_['question_id'] for _ in data]).value_counts().max() == 1

        # Convert the dataset to a list of dictionaries with only "index", "input", and "answer" keys
        data = [{'index': i, 'input': _['question'], 'answer': _['answer']}
                for i, _ in enumerate(data)]
        
        if shuffle:
            np.random.shuffle(data)

        # only choose config.n_test samples
        data = data[:config.n_test]
        return data
        
    # Load NQ dataset
    elif config.dataset == 'nq':
        data = datasets.load_dataset("nq_open", split=config.split)
        
        data = [{
            'index': i,
            'input': _['question'],
            'answer': _['answer'][0]
        } for i, _ in enumerate(data)]
        
        if shuffle:
            np.random.shuffle(data)
            
        data = data[:config.n_test]
        return data

    
    else:
        raise NotImplementedError

def parse_response(response):
    response = response.strip()
    return response

def generate_response(prompt, config: Config, pipeline, tokenizer):
    if config.model[:2] == 'gpt':
        completions = models.gpt(
            system_prompt=None,
            prompt=prompt,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=config.n_sample,
            stop=config.stop
        )
        return [parse_response(response) for response in completions]
    
    elif config.model[:5] == 'llama':
        sequences = pipeline(
            prompt,
            num_return_sequences=config.n_sample,
            eos_token_id=tokenizer.eos_token_id,
            max_length=len(tokenizer(prompt)['input_ids'])+config.max_tokens,
            truncation=True
        )
        return [parse_response(response['generated_text'][len(prompt)+1:].split('\n')[0]) for response in sequences]

def get_cls_embeddings(outputs: np.array, model, tokenizer, to_numpy=True) -> torch.Tensor:
    """
    Calculate the [CLS] token embeddings from the last layer of the model for multiple outputs.

    Args:
        outputs: Array of strings with shape (batch_size,)
        model: The pre-trained model
        tokenizer: The tokenizer associated with the model
        to_numpy: Whether to convert the embeddings to a NumPy array (default: True)

    Returns:
        sentence_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    outputs = outputs.tolist()

    # Move the model to the CPU
    model = model.to('cpu')

    encoded_inputs = tokenizer(outputs, padding=True, truncation=False, return_tensors='pt')

    # Move the encoded inputs to the CPU
    encoded_inputs = {k: v.to('cpu') for k, v in encoded_inputs.items()}

    with torch.no_grad():
        model_output = model(**encoded_inputs)

    # Get the [CLS] token embeddings from the last layer
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]

    if to_numpy:
        sentence_embeddings = sentence_embeddings.cpu().numpy()

    return sentence_embeddings
