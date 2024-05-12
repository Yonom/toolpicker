from lib import grok_api_call, toolpicker_system_prompt, test_data


responses = []
for row in test_data:
    tool = grok_api_call(
        [
            {
                "role": "system",
                "content": toolpicker_system_prompt,
            },
            {"role": "user", "content": row["content"]},
        ]
    )

    responses.append(tool)

import json

with open("dataset/test_output_base.json", "w") as f:
    json.dump(responses, f)
