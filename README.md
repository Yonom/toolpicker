Fine tuned Llama as planner / tool picker (based on Llama 3 8B Instruct)

User Query → **ToolPicker** → [selected strategy to answer query]

→ use Llama 70b Model to execute the selected strategy

Implemented Strategies:

- `answer`: Directly answer the query
- `chain-of-thought`: Think step by step, then answer.
- `web-search`: Search the web for results

Policy: Use the least costly strategy (time, money) to fulfill the user request.

Dataset:

MMLU [high_school + college, mathematics + physics + computer_science]

→ 50 % fine tune set [470 entries]

→ 50 % eval set [471 entries]

Fine tuning dataset:

- System prompt: 'Plan the next action. Options:\n"answer": Directly answer the question.\n"chain-of-thought": Think step by step and answer.\n"web-search": Use a search engine to find the answer.’
- Input: question + choices from MMLU eval set
- Output: `"answer" | "chain-of-thought" | "web-search"`
    → for each question, identify the least costly strategy that yields the correct result
    

Eval Results:

- Always use `answer`: 303/431
- Always use `chain-of-thought`: **348/431**
- Llama 3 70B Instruct: 318/431
- Llama 3 8B Instruct: TODO
- Toolpicker: 318/431