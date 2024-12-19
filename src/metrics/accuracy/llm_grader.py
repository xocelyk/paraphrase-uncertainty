from src.models import gpt


def parse_response(response: str) -> str:
    """
    Parse a response to a string.
    """
    return response.strip().lower()

def grade_response(response: str) -> bool:
    """
    Grade a response as correct or incorrect.
    """
    return response == 'yes'

def get_response(prompt: str) -> str:
    """
    Grade a response using a GPT-3.5 model.
    Use a temperature of 0 and 1 sample.
    """
    n_sample = 1
    temperature = 0
    model = 'gpt-3.5-turbo'
    raw_response = gpt(system_prompt=None, prompt=prompt, model=model, n=n_sample, temperature=temperature)[0]
    return parse_response(raw_response)

def calculate_llm_grader_accuracy(question, response, answer):
    prompt = '''You are a teacher grading a quiz. You are given a question, a reference answer, and predicted answer. Your task is to determine if the predicted answer is correct.
    The predicted answer does not need to be the exact same as the correct answer. The predicted answer can be phrased differently than the reference answer. What matters is that the meaning is the same. Use your best judgment. Respond "Yes" if the predicted answer is correct, "No" otherwise.

    Question: What country is the state of California located in?
    Reference: United States
    Answer: U.S.A.
    Correct: Yes
    
    Question: What is the name of Ernest Hemingway's first novel?
    Reference: The Sun Also Rises
    Answer: The Sun Rises, Too
    Correct: No
    
    Question: {}
    Correct answer: {}
    Predicted answer: {}
    Correct:'''.format(question, answer, response)
    
    response = get_response(prompt)
    return grade_response(response)
