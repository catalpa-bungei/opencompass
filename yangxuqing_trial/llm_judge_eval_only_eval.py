from mmengine.config import read_base
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.evaluator import CascadeEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# Import your judge model configuration
with read_base():
    from opencompass.configs.models.qwen2_5.api_qwen_2_5_vl_72b_instruct import (
        models as judge_model,
    )
    from opencompass.configs.models.qwen3.api_qwen_3_32b import (
        models as inference_model,
    )
    from opencompass.configs.datasets.mmlu.mmlu_gen import \
        mmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.agieval.agieval_gen import \
        agieval_datasets  
    from opencompass.configs.datasets.gpqa.gpqa_gen import \
        gpqa_datasets  # noqa: F401, F403

dataset_type = 'mmlu'

if dataset_type == 'mmlu':
    datasets = mmlu_datasets
    test_range = '[0:5]'
elif dataset_type == 'agieval':
    datasets = agieval_datasets
    test_range = '[0:10]'
elif dataset_type == 'gpqa':
    datasets = gpqa_datasets
    test_range = '[0:100]'


dataset_names = []
for i in range(len(datasets)):
    dataset_names.append(datasets[i]['name'])
print('dataset_names: ', dataset_names)

# Define your judge template
JUDGE_TEMPLATE = """
Please evaluate whether the following response correctly answers the question.
Reference Answer: {obj_gold}
Model Response: {prediction}

Is the model response correct? If correct, answer "<True>"; if incorrect, answer "<False>".
""".strip()

# Dataset reader configuration
# mmlu_reader_cfg = dict(
#     input_columns=['input', 'A', 'B', 'C', 'D'],
#     output_column='target',
#     train_split='dev')

# eval_reader_cfg = dict(input_columns=['origin_prompt','prediction','gold'], output_column='correctness')
agieval_reader_cfg = dict(
    input_columns=['question', 'options', 'gold'], output_column='label', train_split='test', test_range=test_range)
mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    # train_split='dev',
    train_split='test',
    test_range=test_range
)
gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    test_range=test_range
    )

mmlu_path = '/root/.cache/opencompass/data/mmlu/test/'
mmlu_surfix = '_test.csv'
agieval_path = '/root/.cache/opencompass/data/agieval/test/'
agieval_surfix = '.jsonl'
gpqa_path = '/root/.cache/opencompass/data/gpqa/'
gpqa_surfix = 'gpqa_diamond.csv'

if dataset_type == 'mmlu':
    dataset_reader_cfg = mmlu_reader_cfg
    dataset_path = mmlu_path
    dataset_surfix = mmlu_surfix
elif dataset_type == 'agieval':
    dataset_reader_cfg = agieval_reader_cfg
    dataset_path = agieval_path
    dataset_surfix = agieval_surfix
elif dataset_type == 'gpqa':
    dataset_reader_cfg = gpqa_reader_cfg
    dataset_path = gpqa_path
    dataset_surfix = gpqa_surfix

# Inference configuration for the model being evaluated
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AGIEvalDataset_v2, AGIEvalEvaluator
from opencompass.datasets import GPQADataset, GPQA_Simple_Eval_postprocess, GPQAEvaluator
from opencompass.utils.text_postprocessors import match_answer_pattern
from opencompass.utils.text_postprocessors import first_option_postprocess, first_capital_postprocess_multi
# Define a rule-based evaluator
acc_evaluator = dict(type=AccEvaluator)
agi_evaluator = dict(type=AGIEvalEvaluator)
gpqa_evaluator = dict(type=GPQAEvaluator)
gpqa_pred_postprocessor = dict(type=GPQA_Simple_Eval_postprocess)

agieval_cloze_sets = ['gaokao-mathcloze', 'math']
agieval_single_choice_sets = [    'gaokao-chinese',    'gaokao-english',    'gaokao-geography',    'gaokao-history',    'gaokao-biology',    'gaokao-chemistry',    'gaokao-mathqa',    'logiqa-zh',    'lsat-ar',    'lsat-lr',    'lsat-rc',    'logiqa-en',    'sat-math',    'sat-en',    'sat-en-without-passage',    'aqua-rat',]
agieval_multiple_choices_sets = [    'gaokao-physics',    'jec-qa-kd',    'jec-qa-ca',]

if dataset_type == 'mmlu':
    rule_evaluator = acc_evaluator
    dataset_pred_postprocessor = dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])')
elif dataset_type == 'agieval':
    rule_evaluator = agi_evaluator
    dataset_pred_postprocessor = dict(type=first_option_postprocess, options='ABCDE')
elif dataset_type == 'gpqa':
    rule_evaluator = gpqa_evaluator
    dataset_pred_postprocessor = gpqa_pred_postprocessor

# construct eval_cfgs
eval_cfgs = []

for i in range(len(agieval_datasets)):
    name = dataset_names[i]
    if dataset_type == 'agieval':
        if name in agieval_cloze_sets:
            rule_evaluator = agi_evaluator
        if name in agieval_single_choice_sets:
            rule_evaluator = acc_evaluator
            dataset_pred_postprocessor = dict(type=first_option_postprocess, options='ABCDE')
        if name in agieval_multiple_choices_sets:
            rule_evaluator = acc_evaluator
            dataset_pred_postprocessor = dict(type=first_capital_postprocess_multi, options='ABCDE')


    # Define an LLM judge evaluator
    llm_judge_evaluator = dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            # path='/fs-computility/ai-shen/yangxuqing/post_processing/data/agieval/prompt-v5-inconfidence/Qwen2.5-VL-32B-Instruct/combined/',
            # path = '/root/.cache/modelscope/hub/datasets/opencompass/agieval/test',
            path = dataset_path,
            # file_name='qwen2.5vl-32b-agieval-gen-combined-filtered.jsonl',
            # file_name = name + '.jsonl',
            file_name = name + dataset_surfix,
            reader_cfg=agieval_reader_cfg,
        ),
        judge_cfg=judge_model[0],
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    )

    # Configure cascade evaluator (cascade mode)
    cascade_evaluator = dict(
        type=CascadeEvaluator,
        llm_evaluator=llm_judge_evaluator,
        rule_evaluator=rule_evaluator,
        parallel=False  # Cascade mode
    )

    # For parallel mode, set parallel=True
    parallel_evaluator = dict(
        type=CascadeEvaluator,
        llm_evaluator=llm_judge_evaluator,
        rule_evaluator=rule_evaluator,
        parallel=True  # Parallel mode
    )

    # Use the cascade evaluator in your dataset evaluation config
    eval_cfg = dict(evaluator=parallel_evaluator,
                    pred_postprocessor=dataset_pred_postprocessor)
    eval_cfgs.append(eval_cfg)



# eval_cfgs = []
# for i in range(len(agieval_datasets)):
#     name = dataset_names[i]
#     # Evaluation configuration with LLM judge
#     eval_cfg = dict(
#         evaluator=dict(
#             type=GenericLLMEvaluator,
#             prompt_template=dict(
#                 type=PromptTemplate,
#                 template=dict(
#                     begin=[
#                         dict(
#                             role='SYSTEM',
#                             fallback_role='HUMAN',
#                             prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
#                         )
#                     ],
#                     round=[
#                         dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
#                     ],
#                 ),
#             ),
#             dataset_cfg=dict(
#                 type=CustomDataset,
#                 # path='/fs-computility/ai-shen/yangxuqing/post_processing/data/agieval/prompt-v5-inconfidence/Qwen2.5-VL-32B-Instruct/combined/',
#                 path = '/root/.cache/modelscope/hub/datasets/opencompass/agieval/test',
#                 # file_name='qwen2.5vl-32b-agieval-gen-combined-filtered.jsonl',
#                 file_name = name + '.jsonl',
#                 reader_cfg=agieval_reader_cfg,
#             ),
#             judge_cfg=judge_model[0],
#             dict_postprocessor=dict(type=generic_llmjudge_postprocess),
#         ),
#         pred_role='BOT',
#     )
#     eval_cfgs.append(eval_cfg)

# ------
# Dataset configuration
# datasets = [
#     dict(
#         type=CustomDataset,
#         abbr='agieval',
#         path='path/to/your/dataset',
#         file_name='your_dataset.jsonl',
#         reader_cfg=reader_cfg,
#         infer_cfg=infer_cfg,
#         eval_cfg=eval_cfg,
#     )
# ]
# ------

# datasets = datasets[:1]  # Limiting to the first dataset for testing
for i in range(len(datasets)):
    datasets[i]['reader_cfg']['test_range'] = test_range
    datasets[i]['eval_cfg'] = eval_cfgs[i]
print('datasets evaluation configuration: --------------\n',datasets[0]['eval_cfg'])

# Model configuration for the model being evaluated
# models = [
#     dict(
#         type=TurboMindModelwithChatTemplate,
#         abbr='model-to-evaluate',
#         path='path/to/your/model',
#         # ... other model configurations
#     )
# ]
models = inference_model

# Output directory
work_dir = './outputs/llm_judge_eval'


from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner,SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask, ModelEvaluator
from opencompass.runners import LocalRunner,SlurmRunner
from opencompass.models import TurboMindModelwithChatTemplate
# Inference configuration
infer = dict(
    partitioner=dict(
        type=SizePartitioner,
        max_task_size = 2000,
        gen_task_coef = 100,
        #num_worker=8

    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
        max_num_workers=128,
    ),
    task=dict(type=OpenICLInferTask),
)

# Evaluation configuration
eval = dict(
    partitioner=dict(
        type=NaivePartitioner, n=8
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(
            type=OpenICLEvalTask,
            )
    ),
)


