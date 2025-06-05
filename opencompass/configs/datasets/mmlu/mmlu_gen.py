from mmengine.config import read_base

with read_base():
    from .mmlu_openai_simple_evals_gen_b618ea import mmlu_datasets  # noqa: F401, F403

for i in range(len(mmlu_datasets)):
    mmlu_datasets[i]['reader_cfg']['test_range'] = '[0:5]'