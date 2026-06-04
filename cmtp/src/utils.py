from typing import List, Sequence
from datetime import datetime
import zoneinfo
import torch
import zlib
import json
import base64
import random
import numpy as np
import joblib
import os

NY = zoneinfo.ZoneInfo("America/New_York")


def current_time():
    return f'{datetime.now(NY).strftime("%H%M%S_%d%m%y")}'


def track_gpu_memory(tag=""):
    if torch.cuda.is_available():
        return f"{tag} | Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    else:
        return ""


def chunk_list(data: Sequence, num_chunks: int) -> List[Sequence]:
    """Helper to split a list into roughly equal chunks."""
    k, m = divmod(len(data), num_chunks)
    return [
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)
    ]


def get_encoded_files(base_dir: str):
    """Read all .py/.ipynb files in base_dir and encode their contents."""
    codemap = {}
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if (
                filename.endswith(".py")
                or filename.endswith(".ipynb")
                or filename.endswith(".sh")
            ):
                full_path = os.path.join(root, filename)
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    codemap[full_path] = code

    json_bytes = json.dumps(codemap).encode("utf-8")
    compressed = zlib.compress(json_bytes, level=9)
    return base64.b64encode(compressed).decode("ascii")


def unpack_encoded_files(encoded_str: str):
    """Decode and decompress the encoded string back to the original codemap."""
    compressed = base64.b64decode(encoded_str.encode("ascii"))
    json_bytes = zlib.decompress(compressed)
    codemap = json.loads(json_bytes.decode("utf-8"))
    return codemap


########### COT PROCESSING : EQUAL SEGMENTATION ###########
def equal_segmentation(
    cot_id: List[List[int]], bot_id: int, eot_id: int, mot_id: int, span_length: int
):

    # fmt: off
    
    # add mot to fill the last span. (-x % n) gives the number to add.
    cot_id = [ cot + [mot_id,]*(-len(cot)%span_length) for cot in cot_id]

    persample_spanids = []
    for cot in cot_id:
        num_spans = len(cot) // span_length
        ids = []
        for span_idx in range(1, num_spans + 1):
            ids.extend([span_idx] * span_length)
        persample_spanids.append(ids)

    # add bot and eot to cot
    cot_id = [[bot_id] + cot + [eot_id] for cot in cot_id]
    persample_spanids = [[0,] + sp + [0,] for sp in persample_spanids]

    # fmt: on

    return cot_id, persample_spanids


########### COT PROCESSING : ALTERNATING LENGTHS ###########
# Not used in paper, but can be used to different training setups.
def alternate_segmentation(
    cot_id: List[List[int]], bot_id: int, eot_id: int, mot_id: int, len1: int, len2: int
):

    # fmt: off
    new_cot_id = []
    persample_spanids = []

    patterns = [len1, len2]

    for cot in cot_id:
        ids = []
        span_idx = 1
        
        # Keep adding segments only until we cover the current cot length
        while len(ids) < len(cot):
            # Cycles through patterns: len_1, len_2, len_1, len_2...
            cur_len = patterns[(span_idx - 1) % len(patterns)]
            ids.extend([span_idx] * cur_len)
            span_idx += 1

        # Add mot to fill the last span (minimal padding calculated here)
        pad_len = len(ids) - len(cot)
        padded_cot = cot + [mot_id] * pad_len

        # Add boundaries
        new_cot_id.append([bot_id] + padded_cot + [eot_id])
        persample_spanids.append([0] + ids + [0])

    cot_id = new_cot_id

    # fmt: on
    return cot_id, persample_spanids


def seed_everything(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_small_llama(
    save_dir: str,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    num_layers: int = 2,
):
    """Create a small Llama model for local testing by truncating the layer stack.

    Run on a machine with GPU and HF access (needs ~3GB to load the full 1B before truncating).
    Requires: pip install transformers torch
    HF token needed if the repo is gated:
        huggingface-cli login
        # or set HF_TOKEN env var

    Usage:
        create_small_llama('.../model_store/smallama')
        # then load normally:
        #   AutoModelForCausalLM.from_pretrained('.../model_store/smallama')
    """
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.model.layers = nn.ModuleList(model.model.layers[:num_layers])
    model.config.num_hidden_layers = num_layers

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


# To decode and inspect a saved code snapshot:
#   with open("<job_dir>/codesnapshot.txt", "r") as f:
#       codesnapshot = f.read()
#   unpacked_files = unpack_encoded_files(codesnapshot)
#   print(unpacked_files.keys())                  # list all captured files
#   print(unpacked_files["<path/to/file>"])        # print a specific file
