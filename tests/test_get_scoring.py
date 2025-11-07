import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
import torch
import json
import glob
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

cost_model = AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-cost', torch_dtype=torch.bfloat16, device_map='auto')
cost_model_tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/beaver-7b-v1.0-cost')

cost_user_prefix = "BEGINNING OF CONVERSATION: USER: "
cost_assistant_prefix = " ASSISTANT:"


DATASET_NAMES = { 
    'safe_rlhf': "PKU-Alignment/PKU-SafeRLHF", 
}

MODEL_NAMES = { 
    'vicuna_7B': 'lmsys/vicuna-7b-v1.5', 
    'falcon_7B': 'tiiuae/falcon-7b-instruct', 
    'falcon_E_3B': 'tiiuae/Falcon-E-3B-Instruct'
}

DATASET_SUBDIRS = {
    'safe_rlhf': ['data/Alpaca-7B', 'data/Alpaca2-7B', 'data/Alpaca3-8B', 'data']
}

DATASET_NAME = 'safe_rlhf'
MODEL_NAME = 'vicuna_7B'

DATASET = DATASET_NAMES[DATASET_NAME]
MODEL = MODEL_NAMES[MODEL_NAME]
DATASET_SUBDIR = DATASET_SUBDIRS[DATASET_NAME]

DEVICE = 'cuda:0'

if MODEL_NAME == 'vicuna_7B':
    user_prefix = "USER: "
    assistant_prefix = " ASSISTANT:"
elif MODEL_NAME == 'falcon_7B' or 'falcon_E_3B':
    user_prefix = "<|prompter|>"
    assistant_prefix = "<|endoftext|><|assistant|>"


BATCH_SIZE = 16
STEP_INTERVAL = 4
def get_chunk_number(file_path):
    match = re.search(r'_chunk_(\d+)\.json$', file_path)
    if match:
        return int(match.group(1)) 
    return 0 

response_root_dir = "/root/autodl-tmp/output/"

search_pattern = os.path.join(
    response_root_dir, 
    f"{MODEL_NAME}_{DATASET_NAME}_responses_chunk_*.json"
)

response_paths = glob.glob(search_pattern)

sorted_response_paths = sorted(response_paths, key=get_chunk_number)
all_dataset_scores_list = []

with torch.no_grad():
    for chunk_idx, response_path in enumerate(sorted_response_paths, 1):
        if chunk_idx <= 138:
            continue
        score_output_path = os.path.join(
            response_root_dir, 
            f"{MODEL_NAME}_{DATASET_NAME}_costs_chunk_{chunk_idx}.pth"
        )

        print(f"\n--- Processing {response_path} ---")
        with open(response_path, 'r', encoding='utf-8') as f:
            response_list = json.load(f)

        all_stepwise_scores = []  

        for sample_idx, sample in enumerate(tqdm(response_list, desc=f"Chunk {chunk_idx} samples")):
            prompt = sample["input_prompt"]
            gen_tokens = sample["responses_split"]

            if not gen_tokens:
                all_stepwise_scores.append(torch.zeros(0))
                continue

            gen_len = len(gen_tokens)
            if gen_len == 0:
                all_stepwise_scores.append(torch.zeros(0))
                continue
            
            sample_indices = list(range(0, gen_len, STEP_INTERVAL))
            if sample_indices[-1] != gen_len - 1:
                sample_indices.append(gen_len - 1)  # 确保最后一个token必计算
            
            prefix_texts = []
            for t in sample_indices:
                response_prefix = "".join(gen_tokens[: t + 1])
                text = f"{cost_user_prefix}{prompt}{cost_assistant_prefix}{response_prefix}"
                prefix_texts.append(text)
            
            sample_scores = []
            for i in range(0, len(prefix_texts), BATCH_SIZE):
                batch_texts = prefix_texts[i:i + BATCH_SIZE]
                inputs = cost_model_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                ).to(cost_model.device)
            
                outputs = cost_model(**inputs)
                batch_scores = outputs.end_scores.squeeze().detach().cpu()
            
                if batch_scores.dim() == 0:
                    sample_scores.append(batch_scores)
                else:
                    sample_scores.extend(list(batch_scores))
            
            step_scores = torch.zeros(gen_len)
            for idx in range(len(sample_indices)):
                start = sample_indices[idx]
                end = sample_indices[idx + 1] if idx + 1 < len(sample_indices) else gen_len
                step_scores[start:end] = sample_scores[idx]
            
            assert len(step_scores) == gen_len
            all_stepwise_scores.append(step_scores)

        torch.save(all_stepwise_scores, score_output_path)
        print(f"Saved stepwise cost scores to {score_output_path}")