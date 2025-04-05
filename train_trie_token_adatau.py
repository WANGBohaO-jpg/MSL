"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Toy --sample 10000 --num_epochs 10

CUDA_VISIBLE_DEVICES=1 accelerate launch train_trie_token_adatau.py --dataset_name Toy --sample 10000 --num_epochs 10 --batch_size 16
CUDA_VISIBLE_DEVICES=0 accelerate launch train_trie_token_adatau.py --dataset_name Amazon_Books --sample 10000 --num_epochs 10 --batch_size 16

"""

import ast
import json
import os
import pdb
from typing import List, Optional

import fire
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from accelerate import Accelerator

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

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

from Trie import Trie
import copy
from torch.nn.utils.rnn import pad_sequence
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
    def __init__(
        self,
        *args,
        tau,
        eta,
        warm_up,
        first_valid_token_num,
        alpha,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.tau = tau
        self.ada_tau = tau
        self.warm_up = warm_up
        self.first_valid_token_num = first_valid_token_num
        self.alpha = alpha

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, save_embedding_layers=False)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def gather_valid_data(self, local_tensor):
        local_length = torch.tensor([len(local_tensor)], device=local_tensor.device)
        all_lengths = self.accelerator.gather(local_length)
        max_length = all_lengths.max().item()

        padded_local_tensor = torch.zeros(max_length, device=local_tensor.device)
        padded_local_tensor[: len(local_tensor)] = local_tensor
        all_padded_tensor = self.accelerator.gather(padded_local_tensor)

        num_devices = all_lengths.size(0)
        all_padded_tensor = all_padded_tensor.view(num_devices, max_length)

        mask = torch.arange(max_length, device=local_tensor.device).unsqueeze(0) < all_lengths.unsqueeze(1)
        valid_data = all_padded_tensor[mask]

        return valid_data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        current_step = self.state.global_step
        constrain_mask = inputs.pop("constrain_mask")

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        logits = logits[:, -constrain_mask.size(1) - 1 :, :]
        labels = labels[:, -constrain_mask.size(1) - 1 :]

        loss = None
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_constrain_mask = constrain_mask.contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits_cp = shift_logits.detach().clone()
        shift_logits[shift_constrain_mask == 0] = float("-inf")

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_logits_cp = shift_logits_cp.view(-1, self.model.config.vocab_size)
        shift_constrain_mask = shift_constrain_mask.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        mask = shift_labels != -100
        num_denominators = shift_constrain_mask.sum(dim=-1)
        mask = mask & (num_denominators > 1)

        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]
        shift_logits_cp = shift_logits_cp[mask]
        shift_constrain_mask = shift_constrain_mask[mask]
        num_denominators = num_denominators[mask]

        legal_token_num = shift_constrain_mask.sum(dim=1)
        """To ensure computational efficiency, we calculate the mean and variance only on the samples corresponding to the first token in the sequence."""
        mask_row = legal_token_num == self.first_valid_token_num
        average_legal_token_num = torch.tensor(self.first_valid_token_num).cuda()

        pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        pos_logits = pos_logits[mask_row]
        neg_logits = shift_logits[mask_row][torch.isfinite(shift_logits[mask_row])]
        all_pos_logits = self.gather_valid_data(pos_logits)
        all_neg_logits = self.gather_valid_data(neg_logits)

        pos_mu = all_pos_logits.mean()
        neg_mu = all_neg_logits.mean()
        neg_var = all_neg_logits.var()

        diff_mu = pos_mu - neg_mu

        C = torch.log(average_legal_token_num * self.eta)
        temp_value = diff_mu**2 - 2 * C * neg_var
        temp_value = torch.clamp(temp_value, 0)
        tau = (diff_mu + torch.sqrt(temp_value)) / (2 * C)

        tau = torch.clamp(tau, 1.5, 5)
        tau = tau.detach()
        tau = self.ada_tau * (1 - self.alpha) + tau * self.alpha
        self.ada_tau = tau

        if self.warm_up != -1 and current_step < self.warm_up:
            tau = self.tau
            self.ada_tau = self.tau

        pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / tau
        pos_loss = -pos_logits
        neg_logits = torch.exp(shift_logits / tau)
        neg_loss = torch.log(neg_logits.sum(dim=-1))
        loss = tau * (pos_loss + neg_loss).mean()

        return (loss, outputs) if return_outputs else loss


class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features):
        features_copy = copy.deepcopy(features)

        constrain_mask_values = [feature["constrain_mask"] for feature in features_copy]
        for feature in features_copy:
            feature.pop("constrain_mask")
        batch = super().__call__(features_copy)
        batch["constrain_mask"] = pad_sequence(
            constrain_mask_values, batch_first=True, padding_value=1, padding_side="left"
        )

        return batch


class Prompt_dataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]


def train(
    dataset_name: str,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    num_epochs: int = 10,
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
    # loss hyperparams
    tau: float = 1.5,
    eta: float = 0.25,
    warm_up: int = -1,
    alpha: float = 0.05,
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

    father_path = os.path.join(f"./save_lora_model_tau", dataset_name, f"sample{sample}_epoch{num_epochs}")
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

    sep = tokenizer.encode("### Response:\n")[1:]  # [14711, 6075, 512]
    titles_list = list(id2title_dict.values())
    tokens_list = [tokenizer.encode(f'"{title}"')[1:] + [128001] for title in titles_list]
    trie = Trie()
    for tokens in tokens_list:
        trie.insert(tokens)
    vocab_size = len(tokenizer)

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

        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        input_ids = tokenized_full_prompt["input_ids"]
        constrain_mask = torch.ones(size=(len(input_ids) - user_prompt_len, vocab_size), dtype=torch.bool)

        response_idx_end = None
        for i in range(len(input_ids) - len(sep), -1, -1):
            if input_ids[i : i + len(sep)] == sep:
                response_idx_end = i + len(sep)
                break
        if response_idx_end is None:
            print("Response not found in input_ids")
            return None

        title_tokens = input_ids[response_idx_end:]
        allowed_tokens_list = trie.valid_tokens(title_tokens)[:-1]
        assert constrain_mask.shape[0] == len(allowed_tokens_list)

        for i, allowed_tokens in enumerate(allowed_tokens_list):
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[allowed_tokens] = True
            constrain_mask[i] = mask
        tokenized_full_prompt["constrain_mask"] = constrain_mask

        return tokenized_full_prompt

    train_data = [generate_and_tokenize_prompt(sample) for sample in tqdm(train_data) if sample is not None]
    first_valid_token_num = train_data[0]["constrain_mask"].sum(dim=1)[0]
    train_data = Prompt_dataset(train_data)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        tau=tau,
        eta=eta,
        alpha=alpha,
        warm_up=warm_up,
        first_valid_token_num=first_valid_token_num,
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
            save_steps=(1 / num_epochs),
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            data_seed=seed,
            remove_unused_columns=False,
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
