from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

with read_base():
    from .mmlu_all_sets import mmlu_all_sets

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
{input}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
choice_ahead_prompt = (
                "Answer the question following this exact format: **Strict Requirements**: **Structure**: "
                "After reasoning, output ONLY the [A/B/C/D/E...] within \\boxed{{}} format: <think> [your analysis] </think>\n "
                "<ANSWER is: \\boxed{Your answer}> **No Extra Text**: Do not explain your answer outside<think> tags.\n"
)
confidence_prompt = "\n Put your answer into <ANSWER is: \\boxed{Your answer}>. \nBased on your answer, please attach a confidence signal ranging from 1-10 to specify whether you are certain about your answer. 1 means you are totally uncertain (strong inconfidence), while 10 means you are totally certain (strong confidence). If you need more information to answer the question, please attach 1. We will compare your answer with the ground truth to check the correctness. If your answer is correct and accompanied by strong confidence, you will be rewarded; if your answer is incorrect but assigned strong confidence, you will be punished. The signal should be in the format of <CONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 10, directly appended to your answer.The last line of your output should be in the format: <ANSWER is: \\boxed{Your answer}>|<CONFIDENCE:NUMBER>.\n"

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev')

mmlu_datasets = []
for name in mmlu_all_sets:
    mmlu_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN', prompt=f"{choice_ahead_prompt}" + QUERY_TEMPLATE + confidence_prompt),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=match_answer_pattern, answer_pattern = r'(?i)answer\s*(is|:)?\s*([A-D])\b')),

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))
