## setup

%cd /content/
%rm -rf LLaMA-Factory
!git clone https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
%ls
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers==0.0.25
!pip install .[bitsandbytes]

## train

import json

args = dict(
  stage="sft",                        # do supervised fine-tuning
  do_train=True,
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  dataset_dir="dataset",
  dataset="toolpicker",             # use alpaca and identity datasets
  template="llama3",                     # use llama3 prompt template
  finetuning_type="lora",                   # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir="llama3_lora",                  # the path to save LoRA adapters
  per_device_train_batch_size=2,               # the batch size
  gradient_accumulation_steps=4,               # the gradient accumulation steps
  lr_scheduler_type="cosine",                 # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  save_steps=1000,                      # save checkpoint every 1000 steps
  learning_rate=5e-5,                     # the learning rate
  num_train_epochs=12.0,                    # the epochs of training
  max_samples=500,                      # use 500 examples in each dataset
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  quantization_bit=4,                     # use 4-bit QLoRA
  loraplus_lr_ratio=16.0,                   # use LoRA+ algorithm with lambda=16.0
  use_unsloth=True,                      # use UnslothAI's LoRA optimization for 2x faster training
  fp16=True,                         # use float16 mixed precision training
  overwrite_output_dir=True,
)

json.dump(args, open("train_llama3.json", "w", encoding="utf-8"), indent=2)

%cd /content/LLaMA-Factory/

!llamafactory-cli train train_llama3.json

## infer

from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc

%cd /content/LLaMA-Factory/

args = dict(
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="llama3_lora",            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
  use_unsloth=True,                     # use UnslothAI's LoRA optimization for 2x faster generation
)
chat_model = ChatModel(args)

with open('dataset/test.json') as f:
  data = json.load(f)

responses = []
for entry in data:
  messages = [
    {
      "role": "user",
      "content": entry["instruction"],
    },
  ]
  response = chat_model.chat(messages, entry["system"])[0].response_text
  responses.append(response)

torch_gc()

with open('dataset/test_output.json', 'w') as f:
  json.dump(responses, f)
