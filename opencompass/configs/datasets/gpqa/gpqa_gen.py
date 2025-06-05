from mmengine.config import read_base

with read_base():
    from .gpqa_openai_simple_evals_gen_5aeece import gpqa_datasets  # noqa: F401, F403

for i in range(len(gpqa_datasets)):
    gpqa_datasets[i]['reader_cfg']['test_range'] = '[0:100]'