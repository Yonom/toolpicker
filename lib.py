from datasets import load_dataset, concatenate_datasets, Split
from joblib import Memory
from tavily import TavilyClient
from groq import Groq

memory = Memory("cachedir", verbose=0)
client = Groq("")
tavily = TavilyClient(api_key="")


@memory.cache
def get_data():
    hs_math = load_dataset("cais/mmlu", name="high_school_mathematics", split=Split.ALL)
    hs_physics = load_dataset("cais/mmlu", name="high_school_physics", split=Split.ALL)
    hs_cs = load_dataset(
        "cais/mmlu", name="high_school_computer_science", split=Split.ALL
    )
    c_math = load_dataset("cais/mmlu", name="college_mathematics", split=Split.ALL)
    c_physics_data = load_dataset("cais/mmlu", name="college_physics", split=Split.ALL)
    c_cs = load_dataset("cais/mmlu", name="college_computer_science", split=Split.ALL)

    combined_df = (
        concatenate_datasets(
            [
                hs_math,
                hs_physics,
                hs_cs,
                c_math,
                c_physics_data,
                c_cs,
            ]
        )
        .class_encode_column("subject")
        .train_test_split(
            test_size=0.5, shuffle=True, seed=42, stratify_by_column="subject"
        )
    )

    def get_content(row):
        return (
            "Question: "
            + row["question"]
            + "\n\nChoices:\nA: "
            + row["choices"][0]
            + "\nB: "
            + row["choices"][1]
            + "\nC: "
            + row["choices"][2]
            + "\nD: "
            + row["choices"][3]
        )

    answers = ["A", "B", "C", "D"]
    test_data = [
        {
            "content": get_content(row),
            "answer": answers[row["answer"]],
        }
        for row in combined_df["test"]
    ]
    train_data = [
        {
            "content": get_content(row),
            "answer": answers[row["answer"]],
        }
        for row in combined_df["train"]
    ]
    return test_data, train_data


@memory.cache
def grok_api_call(messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


def grok_answer(content):
    return grok_api_call(
        [
            {
                "role": "system",
                "content": "Answer the multiple choice question. Only output one of A, B, C or D and nothing else.",
            },
            {
                "role": "user",
                "content": content,
            },
        ]
    )


def grok_cot(content):
    res = grok_api_call(
        [
            {
                "role": "system",
                "content": 'Answer the question. Think step by step. At the very end, output "So, the correct answer is A/B/C/D"',
            },
            {
                "role": "user",
                "content": content,
            },
        ]
    )
    research = re.search(r"correct answer is ([A-D])", res)
    if research is None:
        return "-"
    else:
        return research.group(1)


@memory.cache
def tavily_search(content):
    response = tavily.qna_search(query=content[-400:])
    return response


def grok_tavily(content):
    search_answer = tavily_search(content)
    print(search_answer)
    return grok_api_call(
        [
            {
                "role": "system",
                "content": "Answer the multiple choice question. Only output one of A, B, C or D and nothing else.",
            },
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "TVLY",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": search_answer,
                "tool_call_id": "TVLY",
            },
        ]
    )


toolpicker_system_prompt = 'Plan the next action. Options:\n"answer": Directly answer the question.\n"chain-of-thought": Think step by step and answer.\n"web-search": Use a search engine to find the answer.'
