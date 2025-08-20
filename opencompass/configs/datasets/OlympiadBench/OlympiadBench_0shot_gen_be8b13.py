from mmengine.config import read_base
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import OlympiadBenchDataset, OlympiadBenchEvaluator, olympiadbench_postprocess_v2


with read_base():
    from .OlympiadBench_categories import categories

# Create prompter instance for problems
olympiadbench_prompter_cfg = dict(
    type='OlympiadBenchPrompter'
)

olympiadbench_reader_cfg = dict(
    input_columns=[
        'problem', 'language', 'subject', 'question_type', 
        'answer_type', 'is_multiple_answer', 'unit', 'questions'
    ], 
    output_column='solution'
)

math_ahead_prompt = (
                "Answer the question following this exact format: **Strict Requirements**: **Structure**: "
                "After reasoning, output ONLY the [numerical number or expression] within \\boxed{{}} format: <think> [your analysis] </think>\n "
                "<ANSWER is: \\boxed{Your answer}> **No Extra Text**: Do not explain your answer outside<think> tags.\n"
)
# confidence_prompt = '\nBased on your answer, please attach a confidence signal ranging from 1-10 to specify whether you are certain about your answer. 1 means you are totally uncertain (strong inconfidence), while 10 means you are totally certain (strong confidence). If you need more information to answer the question, please attach 1. We will compare your answer with the ground truth to check the correctness. If your answer is correct and accompanied by strong confidence, you will be rewarded; if your answer is incorrect but assigned strong confidence, you will be punished. The signal should be in the format of <CONFIDENCE:NUMBER>, where NUMBER ranges from 1 to 10, directly appended to your answer.\n'
confidence_prompt = "not here!"
# Prompt is not modified here!!!!!
prompt = f"{math_ahead_prompt}" + 'Question: {problem}\nAnswer: ' + confidence_prompt

olympiadbench_datasets = []
for _name in categories:
    olympiadbench_infer_cfg = dict(
        prompt_template=dict(
            type='OlympiadBenchTemplate',
            # template=dict(
            #     round=[
            #         dict(role='HUMAN', prompt=prompt),
            #     ],
            # )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    olympiadbench_eval_cfg = dict(
        evaluator=dict(type=OlympiadBenchEvaluator, version='v2'), 
        pred_postprocessor=dict(type=olympiadbench_postprocess_v2),
    )

    olympiadbench_datasets.append(
        dict(
            type=OlympiadBenchDataset,
            abbr=f'OlympiadBench_{_name}',
            path='opencompass/OlympiadBench',
            name=_name,
            reader_cfg=olympiadbench_reader_cfg,
            infer_cfg=olympiadbench_infer_cfg,
            eval_cfg=olympiadbench_eval_cfg,
        )
    )

del _name
