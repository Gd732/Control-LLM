import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
current_dir = os.getcwd() 
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import json
from utils import *
from tqdm import tqdm


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

if DATASET_NAME == 'safe_rlhf':
    ds = merge_datasets_from_subdirs(DATASET, DATASET_SUBDIR)['train'].select_columns(['prompt'])

if MODEL_NAME == 'vicuna_7B':
    user_prefix = "USER: "
    assistant_prefix = " ASSISTANT:"
elif MODEL_NAME == 'falcon_7B' or 'falcon_E_3B':
    user_prefix = "<|prompter|>"
    assistant_prefix = "<|endoftext|><|assistant|>"

model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

class DataCollactor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        data_batch = []
        input_prompt_batch = []
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
            input_prompt_batch.append(sample['input_prompts'])
        batch_data = self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        batch['input_prompts'] = input_prompt_batch
        return batch 

def tokenize_prompts(batch):
    formatted_prompts = [
        f"{user_prefix}{prompt_text}{assistant_prefix}"
        for prompt_text in batch['prompt']
    ]
    tokenized = tokenizer(
        formatted_prompts,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokenized['input_prompts'] = formatted_prompts
    return tokenized


tokenized_ds = ds.map(tokenize_prompts, batched=True)
data_collactor = DataCollactor(tokenizer)
train_dataloader = DataLoader(tokenized_ds, batch_size=8, collate_fn=data_collactor)

def process_hidden_states(outputs):
    last_hidden_states = []
    for idx, hidden_state in enumerate(outputs.hidden_states): # (batch_size, 
        last_hidden_states.append(hidden_state[-1][:, -1, :])
    return last_hidden_states

generated_responses = [] 
generated_responses_split = []
generated_records = []
all_masks = []
all_last_hidden_states = []

SAVE_INTERVAL = 5
saved_chunk_index = 1
output_dir = "/root/autodl-tmp/output/" 
os.makedirs(output_dir, exist_ok=True) 

for s, batch_encoded_input in enumerate(tqdm(train_dataloader)):
    input_ids = batch_encoded_input['input_ids'].to(DEVICE)
    attention_mask = batch_encoded_input['attention_mask'].to(DEVICE)
    input_prompts = batch_encoded_input['input_prompts']
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, 
                                 output_hidden_states=True, return_dict_in_generate=True, 
                                 max_new_tokens=128)
    
    generated_responses = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    last_hidden_states_list = process_hidden_states(outputs) 
    
    indices = torch.arange(outputs.sequences.shape[1], 
                           device=DEVICE).unsqueeze(0).expand(outputs.sequences.shape[0], 
                                                              outputs.sequences.shape[1])
    ans_mask = indices >= input_ids.shape[1]
    pad_mask = (outputs.sequences != tokenizer.pad_token_id) 
    mask = ans_mask & pad_mask
    
    for seq, mask_single in zip(outputs.sequences, mask):
        token_texts = tokenizer.convert_ids_to_tokens(seq.tolist())
        gen_tokens = [tok for tok, m in zip(token_texts, mask_single.tolist()) if m]
        generated_responses_split.append(gen_tokens)
        
    for input_prompt, response_full, gen_response_split in zip(input_prompts, generated_responses, generated_responses_split):
        generated_records.append({
            "input_prompt": input_prompt,
            "responses_full": response_full,
            "responses_split": gen_response_split
        })
    
    last_hidden_states_tensor = torch.stack(last_hidden_states_list, dim=0) 
    last_hidden_states_tensor = last_hidden_states_tensor.permute(1, 0, 2) # (batch_size, max_new_tokens, hidden_state_dim)
    
    all_last_hidden_states.append(last_hidden_states_tensor.cpu())
    all_masks.append(mask.cpu())
    
    if (s + 1) % SAVE_INTERVAL == 0 or (s + 1) == len(train_dataloader):
        print(f"\n--- step {s+1}，saving {saved_chunk_index} ---")
        
        json_path = os.path.join(output_dir, f"{MODEL_NAME}_{DATASET_NAME}_responses_chunk_{saved_chunk_index}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(generated_records, f, ensure_ascii=False, indent=4)
        
        try:
            states_tensor = torch.cat(all_last_hidden_states, dim=0)
            states_path = os.path.join(output_dir, f"{MODEL_NAME}_{DATASET_NAME}_hidden_states_chunk_{saved_chunk_index}.pth")
            torch.save(states_tensor, states_path)
            # print(f"Hidden states shape: {states_tensor.shape}")
        except Exception as e:
            print(f"[!] cannot concat hidden_states: {e}")
            
        try:
            masks_tensor = torch.cat(all_masks, dim=0)
            masks_path = os.path.join(output_dir, f"{MODEL_NAME}_{DATASET_NAME}_masks_chunk_{saved_chunk_index}.pth")
            torch.save(masks_tensor, masks_path)
            # print(f"Hidden states shape: {states_tensor.shape}")
        except Exception as e:
            print(f"[!] cannot concat masks: {e}")

        generated_responses = []
        generated_responses_split = []
        generated_records = []
        all_last_hidden_states = []
        all_masks = []
        saved_chunk_index += 1


