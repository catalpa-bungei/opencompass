from opencompass.datasets.yangxuqing_dataset import MyDataset, MyDatasetEvaluator, mydataset_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer


JUDGE_TEMPLATE = """
Please evaluate whether the following response correctly answers the question.
Reference Answer: {gold}
Model Response: {prediction}

Is the model response correct? If correct, answer "<True>"; if incorrect, answer "<False>".
""".strip()

mydataset_reader_cfg = dict(
    input_columns=['prediction', 'gold'],
    output_column='judge_result',
)

mydataset_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


mydataset_eval_cfg = dict(
    evaluator=dict(type=MyDatasetEvaluator),
    pred_postprocessor=dict(type=mydataset_postprocess))

mydataset_datasets = [
    dict(
        type=MyDataset,
        path = '/fs-computility/ai-shen/yangxuqing/post_processing/data/agieval/prompt-v5-inconfidence/Qwen2.5-VL-32B-Instruct/combined',
        name = 'qwen2.5vl-32b-agieval-gen-combined-filtered.jsonl',
        reader_cfg=mydataset_reader_cfg,
        infer_cfg=mydataset_infer_cfg,
        eval_cfg=mydataset_eval_cfg
    ),
]