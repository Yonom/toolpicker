from lib import get_data, toolpicker_system_prompt
import json

test_data, train_data = get_data()

test_dataset = [
    {
        "system": toolpicker_system_prompt,
        "instruction": row["content"],
    }
    for row in test_data
]

with open("dataset/test.json", "w") as f:
    json.dump(test_dataset, f)
