"""
Semantic entropy implementation following Kuhn et al. https://arxiv.org/abs/2302.09664
"""

from typing import List

import numpy as np
import torch

import src.models as models


class ClassifyWrapper():
    def __init__(self, model_name='microsoft/deberta-large-mnli') -> None:
        self.model_name = model_name
        self.model, self.tokenizer = models.get_entailment_model(), models.get_entailment_tokenizer()

    def entropy_over_classes(self, classes: List[List[str]]) -> float:
        '''
        Calculates the entropy over a list of classes
        '''
        n = sum([len(cls) for cls in classes])
        # get probability of each class
        probs = [len(cls) / n for cls in classes]
        # calculate entropy
        entropy = -sum([p * np.log2(p) for p in probs])
        return entropy

    def get_semantic_entropy(self, question: str, responses: List[str]) -> float:
        '''
        Implements Algorithm 1 from Kuhn et al. https://arxiv.org/pdf/2302.09664.pdf
        '''
        classes = [[responses[0]]]
        for response_idx in range(1, len(responses)):
            # print(classes)
            response = responses[response_idx]
            lonely = True
            for cls_idx, cls in enumerate(classes):
                # take the first element from the class to make the comparison
                cls_sample = cls[0]
                # add an exact match check to speed things up
                if response == cls_sample:
                    cls.append(response)
                    lonely = False
                    break

                elif self._compare(question, cls_sample, response)['deberta_prediction'] == 1:
                    classes[cls_idx].append(response)
                    lonely = False
                    break
            if lonely:
                classes.append([response])
        entropy = self.entropy_over_classes(classes)
        return entropy
        
    @torch.no_grad()
    def _batch_pred(self, sen_1: list, sen_2: list, max_batch_size=128):
        inputs = [_[0] + ' [SEP] ' + _[1] for _ in zip(sen_1, sen_2)]
        inputs = self.tokenizer(inputs, padding=True, truncation=False)
        input_ids = torch.tensor(inputs['input_ids']).to(self.model.device)
        attention_mask = torch.tensor(inputs['attention_mask']).to(self.model.device)
        logits = []
        for st in range(0, len(input_ids), max_batch_size):
            ed = min(st + max_batch_size, len(input_ids))
            logits.append(self.model(input_ids=input_ids[st:ed],
                                attention_mask=attention_mask[st:ed])['logits'])
        return torch.cat(logits, dim=0)
    
    @torch.no_grad()
    def create_sim_mat_unbatched(self, question, answers):
        semantic_set_ids = {ans: i for i, ans in enumerate(answers)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(answers), len(answers), 3))
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(answers):
            for j, ans_j in enumerate(answers):
                if i == j:
                    continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i, j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i, j) in enumerate(indices):
                # return preds instead of logits
                sim_mat_batch[i, j] = torch.nn.functional.softmax(sim_mat_batch_flat[_], dim=-1)
        return dict(
            mapping=[_rev_mapping[_] for _ in answers],
            sim_mat=sim_mat_batch
        )

    @torch.no_grad()
    def create_sim_mat_batched(self, question, answers):
        unique_ans = sorted(list(set(answers)))
        semantic_set_ids = {ans: i for i, ans in enumerate(unique_ans)}
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros((len(unique_ans), len(unique_ans),3))
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i,j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i,j) in enumerate(indices):
                # return preds instead of logits
                sim_mat_batch[i,j] = torch.nn.functional.softmax(sim_mat_batch_flat[_], dim=-1)
        return dict(
            mapping = [_rev_mapping[_] for _ in answers],
            sim_mat = sim_mat_batch
        )

    @torch.no_grad()
    def _pred(self, sen_1: str, sen_2: str):
        input = sen_1 + ' [SEP] ' + sen_2
        input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.model.device)
        out = self.model(input_ids=input_ids)
        logits = out.logits
        # logits: [Contradiction, neutral, entailment]
        return logits

    @torch.no_grad()
    def pred_qa(self, question:str, ans_1:str, ans_2:str):
        return self._pred(f"{question} {ans_1}", f'{question} {ans_2}')

    @torch.no_grad()
    def _compare(self, question:str, ans_1:str, ans_2:str):
        pred_1 = self._pred(f"{question} {ans_1}", f'{question} {ans_2}')
        pred_2 = self._pred(f"{question} {ans_2}", f'{question} {ans_1}')
        preds = torch.concat([pred_1, pred_2], 0)
        
        # Check if both predictions are entailment (index 2)
        deberta_prediction = 1 if (preds.argmax(1) == 2).all() else 0

        # deberta prediction is 1 if both predictions are entailment
        return {'deberta_prediction': deberta_prediction,
                'prob': torch.softmax(preds,1).mean(0).cpu(),
                'pred': preds.cpu()
            }


def calculate_semantic_entropy(question: str, responses: List[str], wrapper: ClassifyWrapper) -> float:
    return wrapper.get_semantic_entropy(question, responses)
