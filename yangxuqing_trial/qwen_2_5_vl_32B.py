from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import \
        mmlu_datasets  
    from opencompass.configs.datasets.agieval.agieval_gen import \
        agieval_datasets  
    from opencompass.configs.datasets.gpqa.gpqa_gen import \
        gpqa_datasets
    
    from opencompass.configs.models.qwen2_5.api_qwen_2_5_vl_32b_instruct import \
        models as qwen2_5_vl_32b_instruct


datasets = mmlu_datasets
models = qwen2_5_vl_32b_instruct

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