# OpenCrate

OpenCrate is a utility for searching local audio sample libraries by sound similarity. It generates audio embeddings using **CLAP** and uses Meta's **FAISS** vector database for fast, local similarity searching.

-----

## Quick Start (Python 3.9 recommended)

### 1\. Clone the repository

```bash
git clone https://github.com/dburenok/opencrate.git
cd opencrate
```

### 2\. (Optional) Create a virtual environment

This is recommended to isolate project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3\. Install dependencies

```bash
pip install torch torchvision torchaudio tqdm faiss-cpu streamlit laion_clap requests librosa
```

### 4\. Build the index

This command crawls the specified directory and creates the index files. **The initial build can take from minutes to hours,** depending on the library size. Subsequent runs are incremental and faster.

```bash
python build_index.py --master 'path/to/your/Sample Library'
```

-----

## Usage

After building the index, launch the web application.

1.  **Start the web application:**

    ```bash
    streamlit run opencrate.py
    ```

2.  **Perform a search:**

      - In the sidebar, navigate to **Similarity Search**.
      - Drag and drop an audio file (`.wav`, `.aif`, `.flac`, `.mp3`, `.ogg`, etc.) into the upload area.
      - The application will display the most similar samples from your library.

-----

## Index Layout

Index files are created in an `opencrate_index/` directory during the build process.

```
opencrate_index/
├─ index.faiss      # The FAISS vector database
├─ embeddings.bin   # Raw float32 audio embeddings from laion_clap
├─ manifest.csv     # File metadata for incremental updates
└─ bad.txt          # List of files that were overly long (>90s), corrupt, or unreadable
```

-----

## Roadmap

  - Waveform previews
  - Text-based search (e.g., "warm chord loop," "vinyl snare")
  - BPM and key extraction
