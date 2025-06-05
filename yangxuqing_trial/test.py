import json
from openai import OpenAI
import os

# 设置基本参数
client = OpenAI(
    api_key="EMPTY",  # 你的 API 密钥
    base_url="http://localhost:8000/v1"  # 本地部署的 OpenAI 服务
)

def test_connection():
    try:
        # 构造一个简单的问题
        prompt = "1 + 1 等于多少？"

        # 调用 OpenAI API
        response = client.chat.completions.create(
            model="Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0  # 调整温度参数
        )

        # 获取预测的答案
        predicted_token = response.choices[0].message.content.strip()

        print(f"连接成功！模型的回答是: {predicted_token}")
        return True
    except Exception as e:
        print(f"连接失败: {e}")
        return False

# 运行测试
test_connection()