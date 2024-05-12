from lib import get_data, toolpicker_system_prompt, grok_answer, grok_cot, grok_tavily
import json

test_data, train_data = get_data()


finetune_dataset = []

for i, row in enumerate(train_data):
    strategy = row["strategy"]

    chat_completion_answer = grok_answer(row["content"])
    if chat_completion_answer == row["answer"]:
        finetune_dataset.append(
            {
                "system": toolpicker_system_prompt,
                "instruction": row["content"],
                "output": "answer",
            }
        )
        continue

    chat_completion_cot = grok_cot(row["content"])
    if chat_completion_cot == row["answer"]:
        finetune_dataset.append(
            {
                "system": toolpicker_system_prompt,
                "instruction": row["content"],
                "output": "chain-of-thought",
            }
        )

    chat_completion_tavily = grok_tavily(row["content"])
    if chat_completion_tavily == row["answer"]:
        finetune_dataset.append(
            {
                "system": toolpicker_system_prompt,
                "instruction": row["content"],
                "output": "web-searcb",
            }
        )


with open("dataset/train.json", "w") as f:
    json.dump(finetune_dataset, f)
