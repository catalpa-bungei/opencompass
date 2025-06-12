import os
import os.path as osp
from typing import Dict, List, Optional

import mmengine
from datasets import Dataset
from mmengine.config import ConfigDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.registry import (DICT_POSTPROCESSORS, ICL_PROMPT_TEMPLATES,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg
from opencompass.utils.logging import get_logger


class GenericLLMEvaluator(BaseEvaluator):
    """Generic LLM evaluator.

    Arguments:
        prompt_template (ConfigDict): The prompt template for evaluation.
        judge_cfg (ConfigDict): The config for Judge LLM.
        dataset_cfg (ConfigDict): The config for dataset.
        pred_postprocessor (ConfigDict): The config for postprocessor.
        dict_postprocessor (ConfigDict): The config for postprocessor,
            used for evaluation results dict.
    """

    def __init__(
        self,
        prompt_template: ConfigDict,
        judge_cfg: ConfigDict,
        dataset_cfg: Optional[ConfigDict] = None,
        pred_postprocessor: Optional[ConfigDict] = None,
        dict_postprocessor: Optional[ConfigDict] = None,
        keep_predictions: bool = False,
    ) -> None:

        print('GenericLLMEvaluator init---------\n')
        print('prompt_template: ', prompt_template)
        self.logger = get_logger()
        # If judge_cfg is not provided, fall back to the default configuration
        if not judge_cfg:
            self.judge_cfg = self.default_judge_cfg
        else:
            self.judge_cfg = judge_cfg
        self.output_path = ''

        self.prompt_template = ICL_PROMPT_TEMPLATES.build(prompt_template)

        # Build Dataset
        self.dataset_cfg = dataset_cfg
        print('dataset_cfg assigned:------------------------------- ', dataset_cfg)
        assert dataset_cfg is not None, 'dataset_cfg is None'

        self.dict_postprocessor = dict_postprocessor
        self.pred_postprocessor = pred_postprocessor

    def build_inferencer(self, ):
        """Build LLM Inference."""
        output_path = self._out_dir
        self.output_path = f'{output_path}.json'
        out_dir, out_name = osp.split(output_path)
        out_name = f'{out_name}.json'

        self.logger.info(
            f'Set self.output_path to {self.output_path} for current task')
        assert self.output_path is not None, 'output_path is None'

        # Build LLM Inference
        max_out_len = self.judge_cfg.get('max_out_len', None)
        batch_size = self.judge_cfg.get('batch_size', None)

        model = build_model_from_cfg(model_cfg=self.judge_cfg)

        self.inferencer = GenInferencer(
            model,
            max_out_len=max_out_len,
            batch_size=batch_size,
            output_json_filepath=out_dir,
            output_json_filename=out_name,
        )

    def score(
        self,
        predictions,
        references: Optional[List] = None,
        test_set: Optional[Dataset] = None,
    ) -> Dict:
        """Apply to single-model scoring.

        Args:
            predictions: List of model predictions
            references: List of reference answers
            test_set: Optional Dataset containing additional
            context for evaluation
        """
        assert len(predictions) == len(
            references), 'predictions and references must have the same length'

        # -------------- Build Inferencer ----------------
        self.build_inferencer()

        # ---------------- Process Predictions ------------------
        original_predictions = predictions
        # print("predictions before postprocess: -----------------------", original_predictions)
        predictions = self.pred_postprocess(predictions)
        # print("predictions after postprocess: -----------------------", predictions)

        # For Single Round Dialogue
        prediction_dict = {'prediction': predictions, 'obj_gold': references, 'original_prediction': original_predictions}

        # ---------------- Build Dataset for LLM Judge -----------------
        print("dataset_cfg: -----------------------", self.dataset_cfg)
        if self.dataset_cfg:
            dataset = build_dataset_from_cfg(self.dataset_cfg)
            length_test = len(dataset.test)
            print('number of rows in dataset test: -----------------------', len(dataset.test))
            for k, v in prediction_dict.items():
                print(f'Adding {k} to dataset---------\n')
                print(f'Adding {v} to dataset------------------\n')
                if len(v) < length_test:
                    # dataset['test'] = dataset['test'][:len(v)]
                    pass
                print("length of dataset test after modification: -----------------------", len(dataset.test))
                dataset.reader.dataset['test'] = dataset.test.add_column(k, v)
                dataset.reader.input_columns.append(k)

            if references:
                print(f'Adding reference to dataset---------\n')
                print(f'Adding {references} to dataset---------\n')
                dataset.reader.input_columns.append('reference')
                if len(references) < length_test:
                    # dataset.reader.dataset['test'] = dataset.reader.dataset['test'][:len(references)]
                    pass
                print("length of dataset test after modification: -----------------------", len(dataset.test))
                dataset.reader.dataset['test'] = dataset.test.add_column(
                    'reference', references)
        else:
            # Handle test_set in the else branch
            from opencompass.datasets.lmeval import LMEvalDataset

            if test_set is not None:
                # If test_set is provided, use it as the base
                # Ensure necessary columns exist
                print("columns in test_set: -----------------------\n", test_set.column_names)
                print("length of test_set: -----------------------\n", len(test_set))
                print("length of references: -----------------------\n", len(references))
                print("references are: -----------------------\n", references)
                if 'prediction' not in test_set.column_names:
                    test_set = test_set.add_column('prediction', predictions)
                if 'reference' not in test_set.column_names:
                    test_set = test_set.add_column('reference', references)
                # very important, the original 'reference' column seems to be empty
                test_set = test_set.remove_columns('prediction')
                test_set = test_set.add_column('prediction', predictions)
                test_set = test_set.add_column('obj_gold', references)
                test_set = test_set.add_column('original_prediction', original_predictions)

                # Prepare input_columns and data dictionary
                input_columns = test_set.column_names
                print("test_set becomes: -----------------------\n", test_set)
                print("input_columns become: -----------------------\n", input_columns)
                inference_repeat = 1
                if isinstance(predictions[0], list):
                    inference_repeat = len(predictions[0])
                    original_test_set_length = len(predictions)
                # original_predictions is a list of original prediction lists
                # predictions is a list of postprocessed predictions lists
                # we want to make sure each row corresponds to a string, instead of a list
                # expand all the columns to match the length of expanded original_predictions, i.e., multiply by inference_repeat
                if inference_repeat > 1:
                    print("\nInference_repeat is greater than 1, expanding predictions and references")
                    expanded_original_predictions = []
                    expanded_predictions = []
                    expanded_references = []
                    for i in range(original_test_set_length):
                        for j in range(inference_repeat):
                            expanded_original_predictions.append(
                                original_predictions[i][j])
                            expanded_predictions.append(predictions[i][j])
                            expanded_references.append(references[i])
                    expanded_indices = []
                    for i in range(len(test_set)):
                        expanded_indices.extend([i] * inference_repeat)
                    test_set = test_set.select(expanded_indices)
                    test_set = test_set.remove_columns('prediction')
                    test_set = test_set.remove_columns('obj_gold')
                    test_set = test_set.remove_columns('original_prediction')
                    test_set = test_set.add_column('prediction', expanded_predictions)
                    test_set = test_set.add_column('obj_gold', expanded_references)
                    test_set = test_set.add_column('original_prediction', expanded_original_predictions)

                data_dict = {
                    column: test_set[column]
                    for column in test_set.column_names
                }
            else:
                # Original default dataset building logic
                input_columns = list(prediction_dict.keys())
                print("whether references exist: -----------------------\n", references)
                if references:
                    input_columns.append('reference')
                data_dict = prediction_dict.copy()
                if references:
                    data_dict['reference'] = references

            # Create LMEvalDataset
            dataset = LMEvalDataset(
                reader_cfg=dict(
                    input_columns=input_columns,
                    output_column=None,
                    train_split='test',
                ),
                **data_dict,
            )

        dataset.reader.output_column = 'reference'
        retriever = ZeroRetriever(dataset)
        # ----------------- LLM Judge ----------------
        self.inferencer.inference(retriever=retriever,
                                  prompt_template=self.prompt_template)

        output = mmengine.load(self.output_path)
        return self.output_postprocess(output, dataset)

    def pred_postprocess(self, predictions: List) -> Dict:
        if self.pred_postprocessor is None:
            return predictions
        else:
            kwargs = self.pred_postprocessor
            proc = TEXT_POSTPROCESSORS.get(kwargs.pop('type'))
            return [proc(pred, **kwargs) for pred in predictions]

    def output_postprocess(self, output: Dict, dataset=None) -> Dict:
        """Postprocess output by adding necessary statistics or data into
        it."""
        import inspect

        if self.dict_postprocessor is None:
            return output
        else:
            kwargs = self.dict_postprocessor
            proc = DICT_POSTPROCESSORS.get(kwargs.pop('type'))
            sig = inspect.signature(proc)
            if 'dataset' in sig.parameters:
                return proc(output,
                            self.output_path,
                            dataset=dataset,
                            **kwargs)
            else:
                return proc(output, self.output_path, **kwargs)

    @property
    def default_judge_cfg(self):
        from opencompass.models import OpenAISDK

        DEFAULT_JUDGE_CFG = dict(
            type=OpenAISDK,
            path=os.environ['OC_JUDGE_MODEL'],
            key=os.environ['OC_JUDGE_API_KEY'],
            openai_api_base=[
                os.environ.get('OC_JUDGE_API_BASE',
                               'https://api.openai.com/v1/')
            ],
            meta_template=dict(round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ], ),
            query_per_second=16,
            batch_size=1024,
            temperature=0.001,
            tokenizer_path='gpt-4o-2024-05-13',
            verbose=True,
            max_out_len=16384,
            max_seq_len=49152,
        )

        return DEFAULT_JUDGE_CFG
