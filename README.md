# SaferVLM

This is the official repository of the USENIX 2025 paper [Bridging the Gap in Vision Language Models in Identifying Unsafe Concepts Across Modalities](https://arxiv.org/abs/2507.11155).

**Disclaimer:** This repo contains examples of hateful and abusive language. Reader discretion is recommended. This repo is intended for research purposes only. Any misuse is strictly prohibited.

## Setup Environment

```bash
bash setup.sh
conda activate llava
```

Then set up your OpenAI API KEY and Hugging Face Token:

```bash
export OPENAI_API_KEY=sk-xxx
export HF_TOKEN=hf_xxx
```

## UnsafeConcepts Dataset

> ⚠️ **Attention:** This dataset contains unsafe content, so it requires access on Hugging Face.
> To use this dataset, first apply for access [here](https://huggingface.co/datasets/yiting/UnsafeConcepts) and fill in your name, affiliation, and ensure that you will only use it for research or education.

After we grant the access, you can use it as follows.
```
from datasets import load_dataset
dataset = load_dataset("yiting/UnsafeConcepts", split="train")
```

## Measurement

### 1. Obtain VLM-Generated Responses
Take llava-v1.5-7b as an exmaple:

```bash
python measure.py --model_name llava-v1.5-7b --capability perception --response_dir output_test
python measure.py --model_name llava-v1.5-7b --capability alignment --response_dir output_test
python measure.py --model_name llava-v1.5-7b --capability alignment_text_only --response_dir output_test
```

To query all VLMs:
```bash
bash scripts/query_vlms.sh 
```
where the environment is setup indenpendently for each VLM.
- Remeber to fill the working directory (DIR) and the specific capability in scripts/query_vlms.sh

### 2. Evaluation
We use fine-tuned RoBERTa classifiers to classify VLM-generated responses.

We provide these checkpints in huggingface, and they will be automatically downloaded during the evaluation process.

- [perception_classifier](https://huggingface.co/yiting/perception_classifier)
- [alignment_classifier](https://huggingface.co/yiting/alignment_classifier)

```bash
python summarize_measure.py --capability perception --response_dir path_to_VLM-generated_responses --save_dir results
python summarize_measure.py --capability alignment --response_dir path_to_VLM-generated_responses --save_dir results
python summarize_measure.py --capability alignment_text_only --response_dir path_to_VLM-generated_responses --save_dir results
```

## RLHF

### 1. Building Training Datasets

```bash
cd RLHF
python build_training_data.py
```
This will procude training datasets tailored for PPO, SFT, and DPO

### 2. Train
We run the training scripts on a **single** A100 (80G). You can choose to do parallel training on multiple GPUs by editing the gpus_per_node, world_size, and batch_size, etc.

```bash
bash scripts/train_ppo.sh
bash scripts/train_sft.sh
bash scripts/train_dpo.sh
```

### 3. Evaluation

```bash
cd ..
python eval_rlhf.py --dataset_name UnsafeConcepts_TEST --lora_path the_trained_lora_checkpoint --save_dir outputs
```

Aggregate all evaluations on alignment capability, general capabilities, and generalization to other datasets:

```bash
bash scripts/eval.sh
```

## Citation
If you find this useful in your research, please consider citing:



## Acknowledgements
