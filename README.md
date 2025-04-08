# MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender

![Static Badge](https://img.shields.io/badge/Conference-SIGIR2025-FF8C00?label=Conference)

This is the PyTorch implementation for our SIGIR 2025 paper. 
> Bohao Wang, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Yan Feng, Chun Chen, Can Wang 2025. MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender. [arXiv link](https://arxiv.org/abs/2504.04178v1)

## Dataset
We follow the sliding window data processing method used in [BIGRec](https://arxiv.org/abs/2308.08434). In the paper, for LLM-based recommendation, we sample 10,000 training instances (train_10000.csv) due to limitations in computational resources, while for traditional recommendation models, we use the full dataset (train.csv).

## Training
### Training using Language Modeling Loss (LML)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_tau.py --dataset_name Toy --sample 10000 --num_epochs 10 --tau 1
```

### Training using Masked Softmax Loss (MSL)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token.py --dataset_name Toy --sample 10000 --num_epochs 10 --tau 4.5
```

### Training using Masked Softmax Loss (MSL) + Adaptive Temperature Strategy (ATS)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_trie_token_adatau.py --dataset_name Toy --sample 10000 --num_epochs 10
```

## Inference
Replace `lora_weights_path` with the path to the model checkpoint, for example: `./save_lora_model/Toy/sample10000_epoch10_CT_tau4.5/0/checkpoint-79`. We perform inference using the checkpoint generated at each epoch.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch inference.py --lora_weights_path "xxx" --dataset Toy
```

## Evaluate
Replace `lora_weights_path` with the path to the parent directory of the model checkpoint, for example: `./save_lora_model/Toy/sample10000_epoch10_CT_tau4.5/0/`.
```
python evaluate_batch_match.py --lora_weights_father_path "xxx"
```

## Citation
If you find the paper useful in your research, please consider citing:
```
@misc{wang2025msltokensneedtuning,
      title={MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender}, 
      author={Bohao Wang and Feng Liu and Jiawei Chen and Xingyu Lou and Changwang Zhang and Jun Wang and Yuegang Sun and Yan Feng and Chun Chen and Can Wang},
      year={2025},
      eprint={2504.04178},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2504.04178}, 
}
```
