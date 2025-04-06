import math
import fire
import numpy as np
import os
import json
import re


def batch(list, batch_size):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i : batch_size * (i + 1)]


def find_checkpoint_dirs(root_dir):
    checkpoint_dirs = []
    names = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint"):
            checkpoint_dirs.append(full_path)
            names.append(entry)
    return checkpoint_dirs, names


def find_files_with_prefix(directory, prefix="predict", suffix="CBS.json"):
    matched_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.startswith(prefix) and file_name.endswith(suffix):
            absolute_path = os.path.abspath(file_path)
            matched_files.append((absolute_path, file_name))

    return matched_files


def evaluate(rank_list, topk_list=[1, 5, 10]):
    rank_array = np.array(rank_list)
    NDCG, HR = [], []

    for k in topk_list:
        Hit_num = (rank_array < k).sum()
        HR.append(Hit_num / len(rank_array))

        mask = rank_array < k
        NDCG_num = 1 / np.log2(rank_array[mask] + 2)
        NDCG.append(NDCG_num.sum() / len(rank_array) / (1.0 / math.log2(2)))

    result_dict = dict()
    for i in range(len(topk_list)):
        result_dict["NDCG@" + str(topk_list[i])] = float(NDCG[i])

    for i in range(len(topk_list)):
        result_dict["HR@" + str(topk_list[i])] = float(HR[i])

    return result_dict


def generate_result_file(lora_weights_path, predict_path, predict_file_name):
    match = re.search(r"predict_(.*?)\.json", predict_file_name)
    if match:
        result = match.group(1)
    save_path = os.path.join(lora_weights_path, "final_result_{}_match.json".format(result))

    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists, skip this evaluation")
        return

    f = open(predict_path, "r")
    test_data = json.load(f)
    f.close()

    rank_list = []
    predict_items = [[title[1:-1] for title in top_K_items["predict"]] for top_K_items in test_data]
    target_items = [item["output"][1:-1] for item in test_data]

    rank_list = []
    for predict_list, target in zip(predict_items, target_items):
        try:
            rank = predict_list.index(target)
        except ValueError:
            rank = float("inf")
        rank_list.append(rank)

    if len(predict_items[0]) == 5:
        topk_list = [1, 5]
    elif len(predict_items[0]) == 10:
        topk_list = [1, 5, 10]
    elif len(predict_items[0]) == 1:
        topk_list = [1]

    result_dict = evaluate(rank_list, topk_list=topk_list)

    print(result_dict)
    f = open(save_path, "w")
    json.dump(result_dict, f, indent=4)


def main(lora_weights_father_path: str):
    checkpoint_path_list, _ = find_checkpoint_dirs(lora_weights_father_path)
    checkpoint_path_list.append(lora_weights_father_path)

    for checkpoint_path in checkpoint_path_list:
        predict_path_list = find_files_with_prefix(checkpoint_path)

        for predict_path, predict_file_name in predict_path_list:
            print("Begin to evaluate: ", predict_path, predict_file_name)
            generate_result_file(checkpoint_path, predict_path, predict_file_name)


if __name__ == "__main__":
    fire.Fire(main)
