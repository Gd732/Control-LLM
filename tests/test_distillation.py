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
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import intervented_model.llama as llama
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob
import re
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset

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

class StudentModel(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dims: list[int]
                 ): 
        super(StudentModel, self).__init__()
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            main_path = nn.Sequential(
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim)
            )
            self.blocks.append(main_path)
            if current_dim == h_dim:
                skip_path = nn.Identity()
            else:
                skip_path = nn.Linear(current_dim, h_dim)
            self.projections.append(skip_path)
            current_dim = h_dim
            
        self.output_head = nn.Linear(current_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        for block, projection in zip(self.blocks, self.projections):
            identity = x
            fx = block(x) 
            skip = projection(identity)
            x = self.activation(fx + skip) 

        x = self.output_head(x)
        return x.squeeze(-1)
        
class FeatureDataset(Dataset):
    def __init__(self, state, labels, mask):
        self.state = state
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return (self.state[idx], self.labels[idx], self.mask[idx])



response_root_dir = "/root/autodl-tmp/output/"

def get_chunk_number(file_path):
    match = re.search(r'_chunk_(\d+)\.json$', file_path)
    return int(match.group(1)) if match else 0

search_pattern = os.path.join(
    response_root_dir,
    f"{MODEL_NAME}_{DATASET_NAME}_responses_chunk_*.json"
)
response_paths = sorted(glob.glob(search_pattern), key=get_chunk_number)

train_hidden_states = None
train_masks = None
train_costs = None
all_responses = []

valid_chunks = []
for p in response_paths:
    idx = get_chunk_number(p)
    if all(os.path.exists(os.path.join(response_root_dir, f"{MODEL_NAME}_{DATASET_NAME}_{t}_chunk_{idx}.pth"))
           for t in ["masks", "costs", "hidden_states"]):
        valid_chunks.append(p)
    else:
        continue

print(f"Found {len(valid_chunks)} complete chunks out of {len(response_paths)}")

total_samples_estimate = len(valid_chunks) * 50

train_hidden_states = None
train_masks = None
train_costs = None
all_responses = []

cursor = 0
for response_path in tqdm(valid_chunks, desc="Loading chunks"):
    chunk_idx = get_chunk_number(response_path)
    mask_path = os.path.join(response_root_dir, f"{MODEL_NAME}_{DATASET_NAME}_masks_chunk_{chunk_idx}.pth")
    cost_path = os.path.join(response_root_dir, f"{MODEL_NAME}_{DATASET_NAME}_costs_chunk_{chunk_idx}.pth")
    hidden_state_path = os.path.join(response_root_dir, f"{MODEL_NAME}_{DATASET_NAME}_hidden_states_chunk_{chunk_idx}.pth")

    mask_chunk = torch.load(mask_path, map_location="cpu")
    cost_chunk = torch.load(cost_path, map_location="cpu")
    hidden_state_chunk = torch.load(hidden_state_path, map_location="cpu")

    B, L, D = hidden_state_chunk.shape
    mask_chunk = mask_chunk[:, -L:]

    # pad cost
    padded_costs = torch.zeros((B, L), dtype=torch.bfloat16)
    for i, cost_seq in enumerate(cost_chunk):
        valid_len = min(len(cost_seq), L)
        padded_costs[i, :valid_len] = cost_seq[:valid_len]

    if train_hidden_states is None:
        train_hidden_states = torch.empty((total_samples_estimate, L, D), dtype=torch.bfloat16)
        train_masks = torch.empty((total_samples_estimate, L), dtype=torch.bool)
        train_costs = torch.empty((total_samples_estimate, L), dtype=torch.bfloat16)

    train_hidden_states[cursor:cursor+B] = hidden_state_chunk
    train_masks[cursor:cursor+B] = mask_chunk
    train_costs[cursor:cursor+B] = padded_costs
    cursor += B

    with open(response_path, 'r', encoding='utf-8') as f:
        all_responses.extend(json.load(f))

train_hidden_states = train_hidden_states[:cursor]
train_masks = train_masks[:cursor]
train_costs = train_costs[:cursor]

print(f"Done. Final total samples: {cursor}")
print(f"Hidden states: {train_hidden_states.shape}")
print(f"Masks:         {train_masks.shape}")
print(f"Costs:         {train_costs.shape}")


train_dataset = FeatureDataset(train_hidden_states, train_costs, train_masks)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

def train_epoch(student_model, train_dataloader, optimizer, device, decay_factor=1.0, consistency_weight=0.0):
    student_model.train()  
    total_loss = 0
    count_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch", unit="batch", dynamic_ncols=True)
    
    for batch_input in progress_bar:
        optimizer.zero_grad()
        hidden_states, labels, masks = batch_input
        
        hidden_states = hidden_states.to(device=device, dtype=torch.bfloat16)
        labels = labels.to(device=device, dtype=torch.bfloat16)
        masks = masks.to(device=device)
        
        batch_size, seq_length, hidden_dim = hidden_states.shape
        predictions = student_model(hidden_states.view(-1, hidden_dim)).view(batch_size, seq_length)

        
        gamma = torch.tensor(decay_factor, device=device, dtype=torch.bfloat16) 
        
        elementwise_loss = F.mse_loss(predictions, labels, reduction='none')
        valid_seq_lengths = masks.sum(dim=1).long()
        last_seq_indices = (valid_seq_lengths - 1).clamp(min=0)
        
        t_indices = torch.arange(seq_length, device=device, dtype=torch.bfloat16).unsqueeze(0)
        
        distance_from_end = (last_seq_indices.unsqueeze(1) - t_indices).clamp(min=0)
        weights = torch.pow(gamma, distance_from_end)
        weighted_loss = elementwise_loss * weights
        weighted_loss_masked = weighted_loss * masks.bool()
        num_valid_tokens = masks.sum()
        
        if num_valid_tokens > 0:
            pred_loss = weighted_loss_masked.sum() / num_valid_tokens
        else:
            pred_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        valid_pair_mask = (masks[:, :-1].bool() & masks[:, 1:].bool())
        valid_preds_t = predictions[:, :-1][valid_pair_mask]
        valid_preds_t_next = predictions[:, 1:][valid_pair_mask]
        
        if valid_preds_t.numel() > 0:
            consistency_loss = consistency_weight * F.mse_loss(valid_preds_t, valid_preds_t_next.detach())
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        loss = pred_loss + consistency_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count_batches += 1
        
        progress_bar.set_postfix(loss=loss.item())

    average_loss = total_loss / count_batches
    return average_loss


EPOCH = 300
LR = 5e-3
student_model = StudentModel(
    input_dim=4096, 
    output_dim=1, 
    hidden_dims=[4096, 1024, 256]
).to(device=DEVICE, dtype=torch.bfloat16)

optimizer = optim.Adam(student_model.parameters(), lr=LR)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

loss_dict = []
lr_dict = []

for epoch in range(EPOCH):
    train_loss = train_epoch(student_model, train_dataloader, optimizer, DEVICE)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    loss_dict.append(train_loss)
    lr_dict.append(current_lr)
    log_data = {
        "loss": loss_dict,
        "lr": lr_dict
    }
    with open(f"trained_model_stage_score/training_log_{MODEL_NAME}_{DATASET_NAME}.json", "w") as f:
        json.dump(log_data, f, indent=4)
        
    if (epoch+1) % 20 == 0:
        torch.save(student_model.state_dict(), f'trained_model_stage_score/student_model_{MODEL_NAME}_{DATASET_NAME}_{epoch+1}.pth')



