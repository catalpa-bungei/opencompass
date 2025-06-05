from opencompass.models import OpenAISDK

api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )


models = [
    dict(
        type=OpenAISDK,
        abbr='Qwen3-32B-non-thinking',
        openai_api_base='http://localhost:8432/v1',
        path='Qwen3-32B-non-thinking',
        key='EMPTY',
        max_seq_len=20000,
        max_out_len=8192,
        batch_size=1024*8,
        # temperature=0.9,
        query_per_second=60,
        run_cfg=dict(num_gpus=0),
        extra_body={
        # "top_k": 20, 
        "chat_template_kwargs": {"enable_thinking": False}
        },
        # extra_body={
        # "chat_template_kwargs": str({"enable_thinking": False}),
        # }
    )
]