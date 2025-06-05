import datasets
from .base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
import json

class MyDataset(BaseDataset):

    @staticmethod
    def load(path, name) -> datasets.Dataset:
        file_name = f'{path}/{name}'
        with open(file_name, 'r') as f:
            data = [json.loads(line) for line in f]
        dataset = datasets.Dataset.from_list(data)
        return dataset

class MyDatasetEvaluator(BaseEvaluator):

    def score(self, predictions, references) -> dict:
        pass

def mydataset_postprocess(text: str) -> str:
    pass