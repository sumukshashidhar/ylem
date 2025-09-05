from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
import torch

max_seq_length = 4096

model, tokenizer = FastModel.from_pretrained(
    model_name = "google/gemma-3-270m-it",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = True, # [NEW!] We have full finetuning now!
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma3",
)

dataset = load_dataset("textcleanlm/textclean-2B-raw-cleaned", split = "train", num_proc=64)

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": "Strength: Medium"},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["responses"]}
        ]
    }

dataset = dataset.map(
    convert_to_chatml, num_proc=64
)

# print(dataset[100])

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=64)

print(dataset[100]['text'])

splits = dataset.train_test_split(test_size=0.01, seed=3407)
trains = splits["train"]
evals  = splits["test"]

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = trains,
    eval_dataset = evals, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_ratio=0.05,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 2e-5,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        eval_strategy="steps",
        eval_steps=0.05,
        save_steps=0.05,
        output_dir="outputs/",
        report_to = "wandb", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("gemma-3-ylem-20250904")  # Local saving
tokenizer.save_pretrained("gemma-3-ylem-20250904")

model.push_to_hub("textcleanlm/gemma-3-ylem-20250904") # Online saving
tokenizer.push_to_hub("textcleanlm/gemma-3-ylem-20250904") # Online saving

model.save_pretrained_merged("gemma-3-ylem-20250904", tokenizer, save_method = "merged_16bit")
model.push_to_hub_merged("textcleanlm/gemma-3-ylem-20250904-16bit", tokenizer, save_method = "merged_16bit")