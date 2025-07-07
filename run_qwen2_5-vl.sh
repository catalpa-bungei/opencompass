conda activate opencompass-vllm
export PYTHONPATH=/fs-computility/wangxuhong/yangxuqing/opencompass/
# source ~/.bashrc
# cd /fs-computility/ai-shen/yangxuqing/opencompass
cd /fs-computility/wangxuhong/yangxuqing/opencompass

# python run.py yangxuqing_trial/qwen_2_5_vl_7B.py
# python run.py yangxuqing_trial/qwen_2_5_vl_32B.py
# python run.py yangxuqing_trial/qwen_2_5_vl_72B.py
# python run.py yangxuqing_trial/qwen_3_32B.py
# python run.py yangxuqing_trial/qwen_3_32B_non-thinking.py
# python run.py yangxuqing_trial/llm_judge_eval_only_eval.py -m eval -r /fs-computility/ai-shen/yangxuqing/opencompass/outputs/default/20250518_213703-gpqa-332b-v5-0.7
# python run.py yangxuqing_trial/llm_judge_eval_only_eval.py -m eval -r /fs-computility/ai-shen/yangxuqing/opencompass/outputs/default/20250518_205908

python run.py yangxuqing_trial/llm_judge_eval_2_5_72b_mmlu.py
python run.py yangxuqing_trial/llm_judge_eval_2_5_72b_gpqa.py
python run.py yangxuqing_trial/llm_judge_eval_2_5_72b_agieval.py
