# generate
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

import torch
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union

from transformers.cache_utils import Cache, DynamicCache


class Intervented_LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def set_opt_control_params(self, student_model, lr, epochs):
        self.student_model = student_model.to(torch.float16)
        self.lr = lr
        self.epochs = epochs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # intervention
        if self.student_model is not None and self.lr is not None and self.epochs is not None:
            hidden_states_opt = hidden_states.detach().clone()
            hidden_states_opt.requires_grad = True
            # with torch.no_grad():
            #     student_out_before = self.student_model(hidden_states).mean().item()

            optimizer = torch.optim.SGD([hidden_states_opt], lr=self.lr)

            with torch.enable_grad():
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    value = self.student_model(hidden_states_opt)   # [B, 1]
                    loss = value.sum()
                    loss.backward()
                    optimizer.step()
            # with torch.no_grad():
            #     student_out_after = self.student_model(hidden_states_opt).mean().item()
            hidden_states = hidden_states_opt.detach().clone()
            # print(f"[Intervention] student out before: {student_out_before:.4f}, after: {student_out_after:.4f}")

        # if self.student_model is not None and self.lr is not None and self.epochs is not None:
        #     hidden_states_opt = hidden_states.detach().clone()
        #     hidden_states_opt.requires_grad = True
        
        #     with torch.no_grad():
        #         student_out_before = self.student_model(hidden_states)
        #         active_mask = (student_out_before > 0)
        
        #     optimizer = torch.optim.SGD([hidden_states_opt], lr=self.lr)
        #     if active_mask.any():
        #         with torch.enable_grad():
        #             for _ in range(self.epochs):
        #                 optimizer.zero_grad()
        #                 value = self.student_model(hidden_states_opt)  # [B, seq, 1]
        #                 active_mask = active_mask & (value >= 0)
        #                 if not active_mask.any():
        #                     break
        #                 broadcast_mask = active_mask.view(-1, 1, 1).float()
        #                 loss = (value * broadcast_mask).sum()
        #                 loss.backward()
        #                 optimizer.step()
        #     hidden_states = hidden_states_opt.detach().clone()
            
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Intervented_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Intervented_LlamaModel(config)

    def set_opt_control_params(self, student_model, lr, epochs):
        self.model.set_opt_control_params(student_model, lr, epochs)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    ds = merge_datasets_from_subdirs(DATASET, DATASET_SUBDIR)['test'].select_columns(['prompt'])

if MODEL_NAME == 'vicuna_7B':
    user_prefix = "USER: "
    assistant_prefix = " ASSISTANT:"
elif MODEL_NAME == 'falcon_7B' or 'falcon_E_3B':
    user_prefix = "<|prompter|>"
    assistant_prefix = "<|endoftext|><|assistant|>"

is_intervent = True

if is_intervent: 
    student_lr = 1e-3
    epochs = 50
    output_dir = "/root/autodl-tmp/output/test/intervent_more" 
    model = Intervented_LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=DEVICE, cache_dir='/root/autodl-tmp/cache/hub')
    student_model = StudentModel(input_dim=4096, output_dim=1, hidden_dims=[4096, 1024, 256]).to(device=DEVICE, dtype=torch.bfloat16)
    student_model.load_state_dict(
        torch.load("trained_model_decay/student_model_vicuna_7B_safe_rlhf_safe_rlhf_200.pth", 
                   map_location=DEVICE))
    model.set_opt_control_params(student_model, student_lr, epochs)
    
else:
    output_dir = "/root/autodl-tmp/output/test/no_intervent" 
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
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
        batch_data = self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        return batch 

def tokenize_prompts(batch):
        formatted_prompts = [
            f"{user_prefix}{prompt_text}{assistant_prefix}" 
            for prompt_text in batch['prompt']
        ]
        return tokenizer(
        formatted_prompts, 
        truncation=True,    
        padding="max_length", 
        max_length=256
    )

tokenized_ds = ds.map(tokenize_prompts, batched=True)
data_collactor = DataCollactor(tokenizer)
train_dataloader = DataLoader(tokenized_ds, batch_size=16, collate_fn=data_collactor)

def process_hidden_states(outputs):
    last_hidden_states = []
    for idx, hidden_state in enumerate(outputs.hidden_states): # (batch_size, 
        last_hidden_states.append(hidden_state[-1][:, -1, :])
    return last_hidden_states

all_generated_responses = [] 
all_masks = []
all_last_hidden_states = []

SAVE_INTERVAL = 5
saved_chunk_index = 1

os.makedirs(output_dir, exist_ok=True) 

for s, batch_encoded_input in enumerate(tqdm(train_dataloader)):
    input_ids = batch_encoded_input['input_ids'].to(DEVICE)
    attention_mask = batch_encoded_input['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, 
                                 max_new_tokens=128)
    
    generated_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    all_generated_responses.extend(generated_response)

    if (s + 1) % SAVE_INTERVAL == 0 or (s + 1) == len(train_dataloader):
        print(f"\n--- step {s+1}，saving {saved_chunk_index} ---")
        
        json_path = os.path.join(output_dir, f"{MODEL_NAME}_{DATASET_NAME}_responses_test_chunk_{saved_chunk_index}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_generated_responses, f, ensure_ascii=False, indent=4)
            

        all_generated_responses = []
        saved_chunk_index += 1