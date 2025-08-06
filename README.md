# OpenCrate

Search your local audio sample library by sound similarity. Uses **CLAP** for audio embeddings and **FAISS** for fast vector search.

---

## Quick Start

**Requires:** Python 3.9

### 1. Clone repo

```bash
git clone https://github.com/dburenok/opencrate.git
cd opencrate
```

### 2. (Optional) Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio tqdm faiss-cpu streamlit laion_clap requests librosa
```

### 4. Build index

Scans your library and creates index files (first run is slow).

```bash
python build_index.py --master 'path/to/your/Sample Library'
```

---

## Usage

Start app:

```bash
streamlit run opencrate.py
```

In the browser:
* Upload an audio file (`.wav`, `.mp3`, `.flac`, etc.)
* View most similar samples.

---

## Index files (`opencrate_index/`)

* **index.faiss** – FAISS database
* **embeddings.bin** – CLAP embeddings
* **manifest.csv** – File metadata
* **bad.txt** – Skipped/corrupt files

---

## Roadmap

* Waveform previews
* Text search (e.g., “warm chord loop”)
* BPM & key detection
