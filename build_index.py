from __future__ import annotations
# ❗️ import torch *before* faiss to avoid segfaults on Apple Silicon
import torch, faiss
import signal, csv, glob, mmap, sys, time
from pathlib import Path
import argparse, os
import numpy as np
from tqdm import tqdm
import requests
import laion_clap
import librosa

# ───────── constants ───────────────────────────────────────────────────────────
DIM          = 512
EXTS         = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")
MANIFEST_CSV = "manifest.csv"
EMB_BIN      = "embeddings.bin"
INDEX_FILE   = "index.faiss"
BAD_FILES    = "bad.txt"
BATCH_SIZE   = 1

# ───────── graceful-interrupt flag ─────────────────────────────────────────────
_stop = False
def _sigint_handler(sig, frame):
    global _stop
    _stop = True
    print("\n⚠️ Interrupt received — finishing current batch then stopping...\n")
signal.signal(signal.SIGINT, _sigint_handler)

# ───────── helpers ─────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_clap():
    """Loads the CLAP music model from a checkpoint."""
    ckpt_url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
    ckpt_path = 'music_audioset_epoch_15_esc_90.14.pt'

    # Download if not present locally
    if not os.path.exists(ckpt_path):
        print(f"Downloading checkpoint to {ckpt_path}...")
        
        try:
            response = requests.get(ckpt_url, stream=True)
            response.raise_for_status()
            with open(ckpt_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the checkpoint: {e}")
            return None

    try:
        device = get_device()
        print(f"Using device: {device}")
        
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')  # base for latest checkpoint
        model.load_ckpt(ckpt_path)
        model = model.to(device)
    
        return model
    except Exception as e:
        print(f"Error loading the CLAP model: {e}")
        return None

def list_audio_files(folder: Path) -> list[Path]:
    files = []

    def scan_dir(path):
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    scan_dir(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    name = entry.name
                    if name.startswith("._"):
                        continue  # skip macOS metadata
                    if Path(name).suffix.lower() in EXTS:
                        files.append(Path(entry.path))

    scan_dir(folder)
    return sorted(files)

def bytes_to_rows(byte_len: int) -> int:
    return byte_len // (DIM * 4)

def is_audio_file_valid(path: Path) -> bool:
    """Check if an audio file is readable, decodable, and within length limits."""
    try:
        # Load audio with native sample rate
        y, sr = librosa.load(str(path), sr=None)

        # Check for invalid or missing sample rate
        if sr is None or sr == 0:
            print(f"Skipping file with invalid sample rate: {path}")
            return False

        # Check for empty waveform
        if y.size == 0:
            print(f"Skipping empty audio file: {path}")
            return False

        # Duration check
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            print(f"Skipping zero-length audio file: {path}")
            return False
        if duration > 90:
            print(f"Skipping audio over 90 seconds in length: {path}")
            return False

        return True

    except Exception:
        print(f"Skipping corrupt or unreadable file: {path}")
        return False

def repair_state(out: Path):
    """Trim .bin/.csv so they're mutually consistent after a crash."""
    bin_f, csv_f = out/EMB_BIN, out/MANIFEST_CSV

    n_bin  = bytes_to_rows(bin_f.stat().st_size) if bin_f.exists() else 0
    n_csv  = sum(1 for _ in csv_f.open())        if csv_f.exists() else 0
    n_keep = min(n_bin, n_csv)

    if n_bin != n_keep:
        with bin_f.open("r+b") as fh:
            fh.truncate(n_keep * DIM * 4)
    if n_csv != n_keep:
        rows = list(csv.reader(csv_f.open()))[:n_keep]
        with csv_f.open("w", newline="") as fh:
            csv.writer(fh).writerows(rows)

def append_batch(bin_f: Path, csv_f: Path, vecs: np.ndarray, paths: list[Path]):
    """Append vectors + metadata atomically *enough* (single-process)."""
    # write embeddings
    with bin_f.open("ab") as fh:
        fh.write(vecs.astype("float32", copy=False).tobytes(order="C"))
        fh.flush(); os.fsync(fh.fileno())

    # write manifest rows
    with csv_f.open("a", newline="") as fh:
        w = csv.writer(fh)
        for p in paths:
            st = p.stat()
            w.writerow([str(p), st.st_size, int(st.st_mtime)])
        fh.flush(); os.fsync(fh.fileno())

# ───────── main build routine ──────────────────────────────────────────────────
def build_index(master: Path, out: Path, batch: int):
    out.mkdir(parents=True, exist_ok=True)
    bin_f, csv_f, idx_f, bad_f = (out/EMB_BIN, out/MANIFEST_CSV, out/INDEX_FILE, out/BAD_FILES)

    # recover from any previous partial-write
    repair_state(out)

    # load manifest and bad_files
    manifest = {row[0] for row in csv.reader(csv_f.open())} if csv_f.exists() else set()
    bad_files = {line.strip() for line in open(bad_f)} if os.path.exists(bad_f) else set()
    
    model = load_clap()

    for pack in sorted((p for p in master.iterdir() if p.is_dir())):
        if _stop:
            break
        
        audio_files = list_audio_files(pack)
        if not audio_files:
            print(f"⏩ {pack.name}: no audio files found")
            continue

        candidate_files = [p for p in audio_files if str(p) not in manifest and str(p) not in bad_files]
        if not candidate_files:
            print(f"⏩ {pack.name}: already indexed ({len(audio_files)} files)")
            continue

        print(f"✔️ {pack.name}: validating {len(candidate_files)} files...")
        new_files = []
        for p in tqdm(candidate_files, ncols=80):
            if _stop:
                break
            if is_audio_file_valid(p):
                new_files.append(p)
            else:
                with open(bad_f, "a") as f:
                    f.write(str(p) + "\n")

        if not new_files:
            print(f"⏩ {pack.name}: no new valid files found to index.")
            continue
        
        print(f"🔍 {pack.name}: indexing {len(new_files)} valid files")
        for i in tqdm(range(0, len(new_files), batch), ncols=80):
            if _stop:
                break
            batch_paths = new_files[i:i+batch]
            vecs = model.get_audio_embedding_from_filelist(x=[str(p) for p in batch_paths])
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            append_batch(bin_f, csv_f, vecs, batch_paths)

        if _stop:
            break

    if _stop:
        print("🛑 Stopped by user. Run the script again to resume.")
        return

    # ── rebuild FAISS index ---------------------------------------------------
    total = bytes_to_rows(bin_f.stat().st_size)
    print(f"🔧 Building FAISS IndexFlatIP: {total:,} vectors")

    with bin_f.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        vecs = np.frombuffer(mm, dtype="float32").reshape(total, DIM).copy()

    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    faiss.write_index(index, str(idx_f))

    print(f"💾 {INDEX_FILE} saved.\n✅ Done!")

# ───────── CLI wrapper ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, type=Path, help="Root folder containing all sample packs")
    ap.add_argument("--out", default="opencrate_index", type=Path, help="Folder to store index files")
    ap.add_argument("--batch", default=BATCH_SIZE, type=int, help="Embedding batch size")
    args = ap.parse_args()

    if not args.master.is_dir():
        sys.exit("❌ Master folder not found.")

    t0 = time.time()
    build_index(args.master, args.out, args.batch)
    print(f"⏱️  Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
