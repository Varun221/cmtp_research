from typing import List, Sequence
from datetime import datetime
import zoneinfo
import torch
import random
import numpy as np
import zlib
import json
import base64
import joblib
import os
import sys

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


def seed_everything(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    codesnapshot = get_encoded_files(project_dir)
    print(codesnapshot[:100])  # Print first 100 characters
    print(sys.getsizeof(codesnapshot) / 1e3)  # Print size in KB

    


