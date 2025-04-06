import ast
import json
import os
from typing import List, Optional

import fire
import pandas as pd
import torch
import transformers
from accelerate import Accelerator

from datasets import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from utils import get_prompt


def generate_list_from_csv(train_data_path, id2title_dict, instuction_str, input_prefix_str):
    def parse_item_ids(item_ids_list):
        titles = [id2title_dict[item_id] for item_id in item_ids_list if item_id in id2title_dict]
        return titles

    df = pd.read_csv(train_data_path)

    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    json_data = []
    for _, row in df.iterrows():
        item_ids_list = row["item_ids"]
        titles = parse_item_ids(item_ids_list)

        input_titles = titles[:-1]
        output_title = titles[-1]

        input_str = input_prefix_str + ", ".join(f'"{title}"' for title in input_titles)
        output_str = f'"{output_title}"'

        json_entry = {"instruction": instuction_str, "input": f"{input_str}\n ", "output": output_str}
        json_data.append(json_entry)

    return json_data


class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, tau=1, loss_type=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        # self.loss_type = loss_type

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss = None
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        mask = shift_labels != -100
        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]

        pos_logits = torch.exp(shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau)
        pos_loss = -torch.log(pos_logits)

        neg_logits = torch.exp(shift_logits / self.tau)
        neg_loss = torch.log(neg_logits.sum(dim=-1))

        loss = (pos_loss + neg_loss).mean()

        return (loss, outputs) if return_outputs else loss


def train(
    dataset_name: str,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    train_on_inputs: int = 0,
    tau: float = 1,
):
    params = locals()
    transformers.set_seed(seed)
    accelerator = Accelerator()

    instruction_prompt, history_prompt = get_prompt(dataset_name)

    id2title_path = os.path.join("./data/", dataset_name, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    train_data_path = os.path.join("./data/", dataset_name, f"train_{sample}.csv")
    train_data = generate_list_from_csv(
        train_data_path=train_data_path,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )

    father_path = os.path.join(
        f"./save_lora_model/",
        dataset_name,
        f"sample{sample}_epoch{num_epochs}_tau{tau}",
    )
    i = 0
    output_dir = os.path.join(father_path, str(i))
    while os.path.exists(output_dir):
        i += 1
        output_dir = os.path.join(father_path, str(i))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size = batch_size // world_size
    gradient_accumulation_steps = batch_size // micro_batch_size // world_size

    """加载了预训练的模型和对应的分词器"""
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]

        return tokenized_full_prompt

    train_data = Dataset.from_list(train_data)
    train_data = train_data.shuffle(seed=seed)
    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        tau=tau,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=0.1,
            save_strategy="steps",
            save_steps=(1 / (num_epochs)),
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            data_seed=seed,
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir, save_embedding_layers=False)


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
