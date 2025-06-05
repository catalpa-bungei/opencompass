from mmengine.config import read_base

with read_base():
    from .agieval_gen_617738 import agieval_datasets  # noqa: F401, F403

for i in range(len(agieval_datasets)):
    agieval_datasets[i]['reader_cfg']['test_range'] = '[0:10]'
