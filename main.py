import argparse
import os
import json
import requests
from refined_seed_patterns import seed_patterns

API_URL = "https://qianfan.baidubce.com/v2/chat/completions"
ACCESS_TOKEN = os.getenv("QIANFAN_API_TOKEN")  # 你需要在环境变量中设置

def call_ernie_model(messages, model):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    payload = {
        "model": model,
        "messages": messages
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise RuntimeError(f"调用失败: {response.status_code} - {response.text}")

    resp_json = response.json()
    return resp_json["result"] if "result" in resp_json else resp_json["choices"][0]["message"]["content"]

def text_rephrase(input_path, output_path, model):
    if not ACCESS_TOKEN:
        raise ValueError("未设置 QIANFAN_API_TOKEN。请设置环境变量。")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    allowed_actions = [k.upper() for k in seed_patterns.keys()]
    results = []

    for idx, line in enumerate(lines):
        prompt = f"""
You are a scientific protocol assistant.

Your job is to rewrite the following chemical procedure as a clear, structured, step-by-step protocol.

🔒 IMPORTANT RULES:
- You MUST use only the following action types as your step starters:
  {', '.join(allowed_actions)}
- The action must appear in ALL CAPS at the beginning of each step.
- Each step should be separated by a semicolon `;`.
- Additional details (reagents, solvents, time, temperature) can follow the action, written in normal English.
- If no chemical action is present, return: NOACTION
- If the text is in a non-English language, return: OTHERLANGUAGE

📄 Original text:
{line}

✍️ Rewritten protocol:
"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        result = call_ernie_model(messages, model)
        results.append(result.strip())

        print(f"✅ 已处理第 {idx + 1} 行")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(item + "\n")

    print(f"\n✅ 所有改写完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逐行改写化学实验步骤")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--model", default="ernie-3.5-8k", help="ERNIE 模型名称（默认 ernie-3.5-8k）")
    args = parser.parse_args()

    text_rephrase(args.input, args.output, args.model)
