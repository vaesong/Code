import re
import os
import json
from pathlib import Path
import itertools
from typing import List, Tuple, Dict, Optional
from typing_extensions import Literal
from collections import defaultdict
import pickle
import tap
import transformers
from tqdm.auto import tqdm
import torch
from torch import nn
from transformers import logging
logging.set_verbosity_error()

from amsolver.environment import Environment
from amsolver.backend.utils import task_file_to_task_class
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
from pyrep.const import RenderMode
from vlm.scripts.cliport_test import CliportAgent
from num2words import num2words

TextEncoder = Literal["bert", "clip"]


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", ignore_mismatched_sizes=True)
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model

def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer

def get_language_feat(language, encoder, num_words, device):
    model = load_model(encoder).to(device)
    tokenizer = load_tokenizer(encoder)
    tokenizer.model_max_length = num_words

    tokens = tokenizer(language, padding="max_length")["input_ids"]
    tokens = torch.tensor(tokens).to(device)

    with torch.no_grad():
        pred = model(tokens).last_hidden_state
    
    return pred