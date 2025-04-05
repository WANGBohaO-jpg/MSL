"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Toy --sample 10000 --num_epochs 10 --seed 43
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Amazon_Books --sample 10000 --num_epochs 10 --seed 43
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Clothing --sample 10000 --num_epochs 10
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Office --sample 10000 --num_epochs 10
"""

import ast
import json
import os
import pdb
import random
import re
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import transformers
from accelerate import Accelerator
from accelerate.utils import gather_object

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
import torch.nn.functional as F
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
        average_legal_token_num,
        tau,
        eta,
        tau_type,
        warm_up,
        dataset_name,
        alpha,
        early_stopping,
        loss_type=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.average_legal_token_num = average_legal_token_num
        self.eta = eta
        self.tau = tau
        self.ada_tau = tau
        self.tau_type = tau_type
        self.warm_up = warm_up  # 用于自适应tau的参数
        self.dataset_name = dataset_name
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.patience = 0

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
        # shift_labels = shift_labels[mask]
        # shift_logits = shift_logits[mask]
        # shift_constrain_mask = shift_constrain_mask[mask]

        num_denominators = shift_constrain_mask.sum(dim=-1)
        mask = mask & (num_denominators > 1)

        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]
        shift_logits_cp = shift_logits_cp[mask]
        shift_constrain_mask = shift_constrain_mask[mask]
        num_denominators = num_denominators[mask]

        """训练使用的masked loss"""
        if self.tau_type == 0:
            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits / self.tau)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = (pos_loss + neg_loss).mean()
        elif self.tau_type == -1:
            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits / self.tau)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = (pos_loss + neg_loss)[num_denominators > 10].mean()
        elif self.tau_type == -2:
            tau0 = self.tau

            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            prob = torch.exp(pos_logits / tau0) / torch.exp(shift_logits / tau0).sum(dim=-1)

            # shift_logits_nan = shift_logits.masked_fill(torch.isinf(shift_logits), float("nan"))
            # mean_logits = torch.nanmean(shift_logits_nan, dim=1)

            # p_tau = torch.exp(shift_logits) / torch.exp(shift_logits).sum(dim=-1, keepdim=True)
            p_tau = torch.exp(shift_logits / tau0) / torch.exp(shift_logits / tau0).sum(dim=-1, keepdim=True)
            E_logits = (shift_logits * p_tau).nansum(dim=-1)
            diff_logits = E_logits - pos_logits
            # diff_logits = mean_logits - pos_logits
            prob_diff_logits = prob * diff_logits

            tau_grad = 2 * (prob.mean() - self.eta) * (prob_diff_logits).sum() / (tau0**2 * pos_logits.size(0))
            tau = tau0 - self.alpha * tau_grad
            tau = torch.clamp(tau, 1, 5)

            all_gpu_tau = self.accelerator.gather(tau)
            # print(all_gpu_tau)
            self.tau = all_gpu_tau.mean().detach().item()

            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits / self.tau)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = (pos_loss + neg_loss).mean()
            self.log({"tau": self.tau, "tau_grad": tau_grad.item(), "prob": prob.mean().item()})
        elif self.tau_type == -3:
            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits_cp)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = (pos_loss + neg_loss)[num_denominators > 10].mean()
        elif self.tau_type == -4:
            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits_cp)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = (pos_loss + neg_loss).mean()
        elif self.tau_type == 2:
            """分析部分，自适应的tau，用合法token数小于某个值的样本"""
            legal_token_num = shift_constrain_mask.sum(dim=1)
            mask_row = legal_token_num < 100
            
            legal_token_num = legal_token_num[mask_row]
            all_legal_token_num = self.gather_valid_data(legal_token_num)
            average_legal_token_num = all_legal_token_num.mean()

            if self.dataset_name == "amazon_game":
                # mask_row = legal_token_num == 50
                # average_legal_token_num = torch.tensor(50).cuda()
                ideal_tau = 1.5
            elif self.dataset_name == "Toy":
                # mask_row = legal_token_num == 46
                # average_legal_token_num = torch.tensor(46).cuda()
                ideal_tau = 4.5
            elif self.dataset_name == "Amazon_Books":
                # mask_row = legal_token_num == 64
                # average_legal_token_num = torch.tensor(64).cuda()
                ideal_tau = 1.5
            elif self.dataset_name == "Clothing":
                # mask_row = legal_token_num == 53
                # average_legal_token_num = torch.tensor(53).cuda()
                ideal_tau = 4

            # 多卡计算
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
            
            self.log({"tau": tau.item()})

            if temp_value == 0:
                ideal_eta = torch.exp(diff_mu / 2 * ideal_tau) / average_legal_token_num
            else:
                ideal_eta = torch.exp(diff_mu / ideal_tau - neg_var / (2 * ideal_tau**2)) / average_legal_token_num

            tau = torch.clamp(tau, 1.5, 5)
            tau = tau.detach()
            tau = self.ada_tau * (1 - self.alpha) + tau * self.alpha
            self.ada_tau = tau

            # 日志可视化
            self.log(
                {
                    "EMA_tau": tau.item(),
                    "neg_var": neg_var.item(),
                    "diff_mu": diff_mu.item(),
                    "temp_value": temp_value.item(),
                    "C": C.item(),
                    "legal_token_num": average_legal_token_num.item(),
                    "ideal_eta": ideal_eta.item(),
                }
            )

            # 自适应tau
            if self.warm_up != -1 and current_step < self.warm_up:
                tau = self.tau
                self.ada_tau = self.tau

            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / tau
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits / tau)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = tau * (pos_loss + neg_loss).mean()
            # loss = (pos_loss + neg_loss).mean()
        elif self.tau_type == 1:
            """分析部分，自适应的tau"""
            legal_token_num = shift_constrain_mask.sum(dim=1)

            if self.dataset_name == "amazon_game":
                mask_row = legal_token_num == 50
                average_legal_token_num = torch.tensor(50).cuda()
                ideal_tau = 1.5
            elif self.dataset_name == "Toy":
                mask_row = legal_token_num == 46
                average_legal_token_num = torch.tensor(46).cuda()
                ideal_tau = 4.5
            elif self.dataset_name == "Amazon_Books":
                mask_row = legal_token_num == 64
                average_legal_token_num = torch.tensor(64).cuda()
                ideal_tau = 1.5
            elif self.dataset_name == "Clothing":
                mask_row = legal_token_num == 53
                average_legal_token_num = torch.tensor(53).cuda()
                ideal_tau = 4
            elif self.dataset_name == "Office":
                mask_row = legal_token_num == 24
                average_legal_token_num = torch.tensor(24).cuda()
                ideal_tau = 3

            # 多卡计算
            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            pos_logits_cp = shift_logits_cp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            probabilities = torch.exp(pos_logits) / torch.exp(shift_logits).sum(dim=-1)
            probabilities_cp = torch.exp(pos_logits_cp) / torch.exp(shift_logits_cp).sum(dim=-1)
            diff_prob = probabilities[mask_row] - probabilities_cp[mask_row]
            diff_prob_mean = diff_prob.mean()
            prob_mean = probabilities[mask_row].mean()

            pos_logits = pos_logits[mask_row]
            neg_logits = shift_logits[mask_row][torch.isfinite(shift_logits[mask_row])]
            all_pos_logits = self.gather_valid_data(pos_logits)
            all_neg_logits = self.gather_valid_data(neg_logits)
            pos_mu = all_pos_logits.mean()
            neg_mu = all_neg_logits.mean()
            neg_var = all_neg_logits.var()

            # pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            # pos_mu = pos_logits[mask_row].mean()
            # neg_mu = shift_logits[mask_row][torch.isfinite(shift_logits[mask_row])].mean()
            # pos_var = torch.var(pos_logits[mask_row])
            # neg_var = torch.var(shift_logits[mask_row][torch.isfinite(shift_logits[mask_row])])
            # pos_mu = pos_logits.mean()
            # neg_mu = shift_logits[torch.isfinite(shift_logits)].mean()
            # pos_var = torch.var(pos_logits)
            # neg_var = torch.var(shift_logits[torch.isfinite(shift_logits)])

            diff_mu = pos_mu - neg_mu

            C = torch.log(average_legal_token_num * self.eta)
            temp_value = diff_mu**2 - 2 * C * neg_var
            temp_value = torch.clamp(temp_value, 0)
            tau = (diff_mu + torch.sqrt(temp_value)) / (2 * C)
            
            self.log({"tau": tau.item()})

            if temp_value == 0:
                ideal_eta = torch.exp(diff_mu / 2 * ideal_tau) / average_legal_token_num
            else:
                ideal_eta = torch.exp(diff_mu / ideal_tau - neg_var / (2 * ideal_tau**2)) / average_legal_token_num

            tau = torch.clamp(tau, 1.5, 5)
            tau = tau.detach()
            tau = self.ada_tau * (1 - self.alpha) + tau * self.alpha
            self.ada_tau = tau

            # 日志可视化
            self.log(
                {
                    "EMA_tau": tau.item(),
                    "neg_var": neg_var.item(),
                    "diff_mu": diff_mu.item(),
                    "temp_value": temp_value.item(),
                    "C": C.item(),
                    "diff_prob_mean": diff_prob_mean.item(),
                    "prob_mean": prob_mean.item(),
                    "legal_token_num": average_legal_token_num.item(),
                    "ideal_eta": ideal_eta.item(),
                }
            )

            # 自适应tau
            if self.warm_up != -1 and current_step < self.warm_up:
                tau = self.tau
                self.ada_tau = self.tau

            pos_logits = shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / tau
            pos_loss = -pos_logits
            neg_logits = torch.exp(shift_logits / tau)
            neg_loss = torch.log(neg_logits.sum(dim=-1))
            loss = tau * (pos_loss + neg_loss).mean()
            # loss = (pos_loss + neg_loss).mean()

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
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    dataset_name: str = "amazon_game",
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
    # train_on_inputs: int = 0,
    # 额外的loss参数
    # loss_type: int = 0,
    tau: float = 1.5,
    eta: float = 0.25,
    tau_type: int = 1,
    warm_up: int = -1,
    alpha: float = 0.05,
    early_stopping: int = 0,
):
    params = locals()
    transformers.set_seed(seed)
    accelerator = Accelerator()

    model_name = re.search(r"Llama[^_]+", base_model).group(0)

    instruction_prompt, history_prompt = get_prompt(dataset_name)

    id2title_path = os.path.join("/c23034/wbh/code/CBS_LLM4Rec/data/", dataset_name, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    train_data_path = os.path.join("/c23034/wbh/code/CBS_LLM4Rec/data/", dataset_name, f"train_{sample}.csv")
    train_data = generate_list_from_csv(
        train_data_path=train_data_path,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )

    father_path = os.path.join(
        f"/c23034/wbh/code/CBS_LLM4Rec/save_lora_model_tau_{model_name}",
        dataset_name,
        f"sample{sample}_epoch{num_epochs}_eta{eta}_alpha{alpha}_tau{tau}_warmup{warm_up}",
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

        # 开始计算constrain mask
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
        # 返回一个和title_tokens长度相同的list，其中每个list元素是对应位置的token space，最后一个是空列表所以删去
        allowed_tokens_list = trie.valid_tokens(title_tokens)[:-1]
        assert constrain_mask.shape[0] == len(allowed_tokens_list)

        for i, allowed_tokens in enumerate(allowed_tokens_list):
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[allowed_tokens] = True
            constrain_mask[i] = mask
        tokenized_full_prompt["constrain_mask"] = constrain_mask

        return tokenized_full_prompt

    train_data = [generate_and_tokenize_prompt(sample) for sample in tqdm(train_data) if sample is not None]
    # 计算平均合法token数量
    constrain_mask_list = [sample["constrain_mask"] for sample in train_data]
    legal_token_num_list = [mask.sum(dim=1) for mask in constrain_mask_list]
    legal_token_num = torch.cat(legal_token_num_list)
    # legal_token_num = legal_token_num[legal_token_num > 1]
    average_legal_token_num = legal_token_num.sum() / legal_token_num.size(0)
    print("average_legal_token_num: ", average_legal_token_num)
    # pdb.set_trace()

    if dataset_name == "amazon_game":
        average_legal_token_num = torch.tensor(1257).to(accelerator.device)
    elif dataset_name == "Toy":
        average_legal_token_num = torch.tensor(883).to(accelerator.device)
    elif dataset_name == "Amazon_Books":
        average_legal_token_num = torch.tensor(1182).to(accelerator.device)

    train_data = Prompt_dataset(train_data)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=val_data,
        data_collator=CustomDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        # loss_type=loss_type,
        average_legal_token_num=average_legal_token_num,
        tau=tau,
        eta=eta,
        tau_type=tau_type,
        alpha=alpha,
        warm_up=warm_up,
        dataset_name=dataset_name,
        early_stopping=early_stopping,
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
            # evaluation_strategy="steps",
            # eval_steps=0.1,
            save_strategy="steps",
            save_steps=(1 / num_epochs),
            # save_steps=(1 / (4 * num_epochs)),
            # save_total_limit=10,
            save_on_each_node=False,
            log_on_each_node=False,
            # load_best_model_at_end=True,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            # ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            data_seed=seed,
            # dataloader_num_workers=2,  # 可能可以设置得更大？
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
