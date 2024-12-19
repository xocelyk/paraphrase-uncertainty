from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config

device = config.device

def get_entailment_model():
    return AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

def get_entailment_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")