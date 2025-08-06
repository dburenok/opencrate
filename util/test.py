from pathlib import Path
import pandas as pd
import numpy as np
import mmap
import laion_clap
import torch

# --- Device ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- Load CLAP ---
def load_clap():
    ckpt_path = '../music_audioset_epoch_15_esc_90.14.pt'
    device = get_device()
    print(f"Using device: {device}")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(ckpt_path)
    model = model.to(device)
    return model

clap_model = load_clap()

# --- Load manifest ---
manifest_path = Path("../opencrate_index/manifest.csv")
manifest = pd.read_csv(manifest_path, header=None)

# --- Load embeddings from .bin ---
bin_f = Path("../opencrate_index/embeddings.bin")
total_rows = bin_f.stat().st_size // (512 * 4)

print(f"Manifest rows: {len(manifest)}, Embedding rows: {total_rows}")
if len(manifest) != total_rows:
    print("❌ Row count mismatch!")

with bin_f.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    stored_vecs = np.frombuffer(mm, dtype="float32").reshape(total_rows, 512).copy()

# --- Compare embeddings ---
for i, row in manifest.iterrows():
    file_path = row.iloc[0]  # assuming first column is file path
    print(f"[{i+1}/{len(manifest)}] Checking: {file_path}")

    path_obj = Path(file_path)
    if not path_obj.exists():
        print(f"  ⚠️ Missing file: {file_path}")
        continue

    # Embed fresh
    new_vec = clap_model.get_audio_embedding_from_filelist([str(path_obj)]).astype("float32")
    new_vec /= np.linalg.norm(new_vec, axis=1, keepdims=True)

    # Stored
    old_vec = stored_vecs[i]

    # Cosine similarity
    sim = old_vec @ new_vec[0]
    print(f"  Cosine similarity: {sim:.6f}")

    if sim < 0.999:
        print("  ❌ Mismatch detected")
    else:
        print("  ✅ Match")
