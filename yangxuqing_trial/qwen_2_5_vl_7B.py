from mmengine.config import read_base

with read_base():
    # from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import \
    #     gsm8k_datasets
    # from opencompass.configs.datasets.demo.demo_math_chat_gen import \
    #     math_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen import \
        mmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.agieval.agieval_gen import \
        agieval_datasets  
    from opencompass.configs.datasets.gpqa.gpqa_gen import \
        gpqa_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.yangxuqing_dataset.yangxuqing_dataset import \
        mydataset_datasets  # noqa: F401, F403
    from opencompass.configs.models.qwen2_5.api_qwen_2_5_vl_7b_instruct import \
        models as qwen2_5_vl_7b_instruct

# datasets = gsm8k_datasets + math_datasets
datasets = mydataset_datasets
models = qwen2_5_vl_7b_instruct

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner,SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
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
            type=OpenICLEvalTask)
    ),
)
