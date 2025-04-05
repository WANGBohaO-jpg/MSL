import ast
import pandas as pd
import fire
import torch

import json
from tqdm import tqdm
import os

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from genre.trie import MarisaTrie
import transformers
from utils import get_prompt


def generate_list_from_csv(data_path, id2title_dict, instuction_str, input_prefix_str):
    def parse_item_ids(item_ids_list):
        titles = [id2title_dict[item_id] for item_id in item_ids_list if item_id in id2title_dict]
        return titles

    df = pd.read_csv(data_path)

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


def main(
    lora_weights_path: str,
    dataset: str,
    test_sample: int = 5000,
    batch_size: int = 8,
    base_model: str = "/c23034/wbh/Llama3_Checkpoints/",
    num_beams: int = 10,
    constraint_BS: int = 1,
):
    transformers.set_seed(42)
    accelerator = Accelerator()
    accelerator.print("Dataset: ", dataset)
    accelerator.print("LoRA Weights Path: ", lora_weights_path)

    data_path = os.path.join("/c23034/wbh/code/CBS_LLM4Rec/data/", dataset)
    test_data_path = os.path.join(data_path, f"test_{test_sample}.csv")
    accelerator.print("test_data_path: ", test_data_path)

    instruction_prompt, history_prompt = get_prompt(dataset)

    id2title_path = os.path.join("/c23034/wbh/code/CBS_LLM4Rec/data/", dataset, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    test_data = generate_list_from_csv(
        data_path=test_data_path,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )

    num_sample = test_sample
    result_json_data = f"predict_{dataset}_{num_sample}_CBS" if constraint_BS else f"predict_{dataset}_{num_sample}_BS"
    result_json_data = os.path.join(lora_weights_path, result_json_data + ".json")
    
    if os.path.exists(result_json_data):
        accelerator.print(f"The {result_json_data} has existed.")
        return
    accelerator.wait_for_everyone()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.bfloat16)
    model.merge_and_unload()
    model.generation_config.cache_implementation = "static"
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    sep = tokenizer.encode("### Response:\n")[1:]  # [14711, 6075, 512]
    titles_list = list(id2title_dict.values())
    tokens_list = [tokenizer.encode("### Response:\n" + f'"{title}"')[1:] for title in titles_list]
    trie = MarisaTrie(tokens_list)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list:
        input_ids = input_ids.tolist()
        for i in range(len(input_ids)):
            if input_ids[i : i + len(sep)] == sep:
                break

        prefix = input_ids[i:]
        allowed_tokens = trie.get(prefix)
        allowed_tokens = [tokenizer.eos_token_id] if allowed_tokens == [] else allowed_tokens

        return allowed_tokens


    def evaluate(instructions, inputs, num_beams=num_beams, max_new_tokens=128):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)

        with torch.no_grad():
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if constraint_BS else None,
            )
        
            sequences_scores = generation_output.sequences_scores.tolist()
            sequences_scores = [
                sequences_scores[i * num_beams : (i + 1) * num_beams] for i in range(len(sequences_scores) // num_beams)
            ]

            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            output = [_.split("Response:\n")[-1] for _ in output]
            real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]

        return real_outputs, sequences_scores

    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"])),
            total=(len(input_temp["instructions"]) + batch_size - 1) // batch_size,
        ):
            instructions, inputs = batch1
            output, sequences_score = evaluate(instructions, inputs)
            outputs.extend(output)
            sequences_scores.extend(sequences_score)

    outputs = gather_object(outputs)
    sequences_scores = gather_object(sequences_scores)
    assert len(outputs) == len(test_data)
    assert len(sequences_scores) == len(test_data)

    if accelerator.is_main_process:
        for i, _ in enumerate(test_data):
            test_data[i]["predict"] = outputs[i]
            test_data[i]["scores"] = sequences_scores[i]

        with open(result_json_data, "w") as f:
            json.dump(test_data, f, indent=4)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
