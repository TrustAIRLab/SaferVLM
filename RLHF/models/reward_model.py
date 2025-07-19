# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput
from transformers import RobertaTokenizer, RobertaModel
    
from peft import PeftModel, LoraModel, LoraConfig

from models.lora_model import get_accelerate_model

from llava.model import *

import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, CLIPProcessor, CLIPModel


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        hidden_size_attr_name = "hidden_size"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None

class SentimentClassifier(torch.nn.Module):
    """Custom classifier using RoBERTa."""
    def __init__(self, model_name="roberta-base", num_labels=3):
        super(SentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]  # Use the representation of the first token ([CLS] token)
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        return logits
    


class RewardModel(torch.nn.Module):
    
    def __init__(self, args: Namespace, **kwargs):
        super(RewardModel, self).__init__()
        self.args = args
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.compute_type = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"
        
        # Load models for both safety reward and similarity reward
        self.tokenizer, self.safety_model = self.load_safety_model()
        
        self.safety_model.to(self.device)
    
    def load_safety_model(self):
        tokenizer = RobertaTokenizer.from_pretrained(self.args.reward_model_base)
        model = SentimentClassifier(self.args.reward_model_base, num_labels=3)
        state_dict = torch.load(self.args.reward_model_name_or_path)
        model.load_state_dict(state_dict, strict=False)
        return tokenizer, model

    def forward(self, text_responses, labels):
        """
        Safety reward model, classifies text responses as 'Safe' or 'Unsafe'.
        """
        label_map = {"Safe": 0, "Unsafe": 1}
        if isinstance(labels, str):
            labels = torch.tensor([label_map[la] for la in labels], dtype=torch.long)
        
        input_ids = self.tokenizer(text_responses, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        
        with torch.cuda.amp.autocast():
            logits = self.safety_model(input_ids, attention_mask)
            # Compute normalized cross entropy loss
            loss = F.cross_entropy(logits, labels, reduction='none')  # Compute loss for each instance in the batch
            loss = -loss  # maximize the reward
            rewards = (loss - loss.min()) / (loss.max() - loss.min())  # Normalization
        return rewards

