from opencompass.datasets import MedmcqaDataset, MedmcqaEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

SYSTEM_PROMPT = 'You are a helpful medical assistant.\n\n' # Where to put this?
ZERO_SHOT_PROMPT = 'Q: {question}\n Please select the correct answer from the options above and output only the corresponding letter (A, B, C, D, or E) without any explanation or additional text.\n'

# Reader configuration
reader_cfg = dict(
    input_columns=[
        'question',
        'options',
        'subject_name',
        'choice_type',
        'prompt_mode',
        'topic_name',
    ],
    output_column='label',
)

# Inference configuration
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(role='SYSTEM', fallback_role='HUMAN', prompt=SYSTEM_PROMPT),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt=ZERO_SHOT_PROMPT, # prompt mode: zero-shot
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
eval_cfg = dict(
    evaluator=dict(type=MedmcqaEvaluator),
    pred_role='BOT',
)
medmcqa_dataset = dict(
    type=MedmcqaDataset,
    abbr='medmcqa',
    path='openlifescienceai/medmcqa',
    prompt_mode='zero-shot',
    reader_cfg=reader_cfg,
    infer_cfg=infer_cfg,
    eval_cfg=eval_cfg,
    
)

medmcqa_datasets = [medmcqa_dataset]
