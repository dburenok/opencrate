from __future__ import annotations
from pathlib import Path
import io, tempfile, numpy as np
import streamlit as st
from datetime import datetime
import mimetypes
import pandas as pd
import requests, os
import laion_clap

# ❗ import torch *before* faiss to avoid segfaults on Apple Silicon
import torch, faiss

# --- Constants ---
INDEX_DIR       = Path("./opencrate_index")
FAISS_FILE      = INDEX_DIR / "index.faiss"
MANIFEST_CSV    = INDEX_DIR / "manifest.csv"
DIM             = 512
AUDIO_EXTS      = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")
TOP_K           = 50
GRID_COLUMNS    = 2

def score_to_color(score: float) -> str:
    """Maps a similarity score (0.2 to 0.8) to an HSL color (red to green)."""
    score = max(0, min(1, score))
    hue = (score - 0.2) / (0.8 - 0.2) * 120
    hue = max(0, min(120, hue))
    return f"hsl({hue}, 90%, 45%)"

st.set_page_config(page_title="OpenCrate | Similarity Search", layout="wide", initial_sidebar_state="expanded")
st.title("OpenCrate Similarity Search")

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --- Cached Resource Loading ---
@st.cache_resource(show_spinner="Loading index…")
def load_faiss() -> tuple[faiss.Index, list[str]]:
    if not FAISS_FILE.exists() or not MANIFEST_CSV.exists():
        st.error(f"Index files not found in `{INDEX_DIR}`. Please ensure 'index.faiss' and 'manifest.csv' are present.")
        st.stop()

    index = faiss.read_index(str(FAISS_FILE))
    try:
        df = pd.read_csv(MANIFEST_CSV, header=None)
        if df.empty:
            st.error(f"'{MANIFEST_CSV}' is empty. Cannot load file paths.")
            st.stop()
        paths = df.iloc[:, 0].astype(str).tolist()
    except Exception as e:
        st.error(f"Error reading manifest file '{MANIFEST_CSV}': {e}")
        st.stop()

    return index, paths

@st.cache_resource(show_spinner="Loading audio engine...")
def load_clap():
    # Using a Hugging Face mirror for the checkpoint
    ckpt_url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
    ckpt_path = 'music_audioset_epoch_15_esc_90.14.pt'

    if not os.path.exists(ckpt_path):
        st.info("Downloading CLAP model checkpoint...")
        try:
            response = requests.get(ckpt_url, stream=True)
            response.raise_for_status()
            with open(ckpt_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading the checkpoint: {e}")
            st.stop()

    try:
        device = get_device()
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        model.load_ckpt(ckpt_path)
        model = model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading the CLAP model: {e}")
        st.stop()


# --- Load Models at App Startup ---
clap_model = load_clap()
faiss_index, path_list = load_faiss()

# --- Cache the FAISS search so it only runs once per query file ---
@st.cache_data(show_spinner="Searching for similar audio...")
def run_similarity_search(file_bytes: bytes) -> list[tuple[str, float]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        vec = clap_model.get_audio_embedding_from_filelist([tmp_path]).astype("float32")
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    finally:
        # Ensure temporary file is always deleted
        Path(tmp_path).unlink(missing_ok=True)

    # Overfetch to handle skips (e.g., missing files)
    extra_k = TOP_K * 5
    distances, indices = faiss_index.search(vec, extra_k)
    similarities = distances[0]
    ids = indices[0]

    playable_results = []
    seen_paths = set()
    for idx, sim in zip(ids, similarities):
        if idx < 0 or idx >= len(path_list): continue # Guard against invalid indices

        path_str = path_list[idx]
        if path_str in seen_paths: continue # Avoid duplicates

        path_obj = Path(path_str)

        # Skip unsupported formats for st.audio and missing files
        if path_obj.suffix.lower() in (".aif", ".aiff"):
            continue
        if not path_obj.exists():
            continue

        playable_results.append((path_str, float(sim)))
        seen_paths.add(path_str)

        if len(playable_results) >= TOP_K:
            break

    return playable_results

# --- Sidebar for UI Controls ---
with st.sidebar:
    st.header("🔍 Search Controls")
    query_file: io.BytesIO | None = st.file_uploader(
        "Find similar samples",
        type=[e.lstrip(".") for e in AUDIO_EXTS],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"✅ Index loaded with **{faiss_index.ntotal:,}** samples.")
    st.caption(f"🗓️ {datetime.now().strftime('%B %-d, %Y')}")

# --- Main Application Logic ---

# 1. Handle file upload and update session state
if query_file is not None:
    query_bytes = query_file.getvalue()
    # Check if this is a new file upload to avoid re-running search on every interaction
    if "last_query_bytes" not in st.session_state or st.session_state.last_query_bytes != query_bytes:
        st.session_state.last_query_bytes = query_bytes
        st.session_state.results = run_similarity_search(query_bytes)
        st.session_state.query_name = query_file.name

# 2. Display results from session state
if "results" not in st.session_state:
    st.info("⬅️ **Upload an audio file in the sidebar to begin.**")
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1.  **Upload:** Drop a `.wav`, `.mp3`, or `.flac` file into the uploader on the left.
    2.  **Search:** The system will analyze your audio and find the most similar sounds from the indexed library.
    3.  **Explore:** Your results will appear below in a grid, ready for you to listen, compare, and download.
    """)

else:
    results = st.session_state.results
    query_name = st.session_state.query_name

    st.header(f"Top {len(results)} Matches for *{query_name}*")
    st.markdown("""Results are displayed in a grid, sorted by similarity score (higher is better).""")

    # --- Create Grid Layout ---
    # Create a new row for every `GRID_COLUMNS` items
    for i in range(0, len(results), GRID_COLUMNS):
        # Get the results for the current row
        row_results = results[i : i + GRID_COLUMNS]
        # Create columns for the row
        cols = st.columns(GRID_COLUMNS)

        for col, result_item in zip(cols, row_results):
            if result_item is None:
                continue

            path_str, sim = result_item
            path_obj = Path(path_str)
            rank = results.index(result_item) + 1

            # Use a container within each column for a "card" effect
            with col:
                # Using the path string as part of the key ensures stability and uniqueness
                # across re-runs, which is crucial for preventing UI glitches.
                with st.container(border=True, key=f"result_{path_str}"):
                    st.audio(str(path_obj))

                    # Truncate long file names for display
                    display_name = (path_obj.name[:30] + '…') if len(path_obj.name) > 30 else path_obj.name
                    st.markdown(f"**{rank}.** {display_name}", help=path_obj.name)

                    color = score_to_color(sim)
                    st.markdown(
                        f'Score: <strong style="color: {color}">{sim:.4f}</strong>',
                        unsafe_allow_html=True
                    )

                    try:
                        with open(path_obj, "rb") as f:
                            data = f.read()

                        mime_type, _ = mimetypes.guess_type(path_obj)
                        if mime_type is None:
                            mime_type = "application/octet-stream"

                        st.download_button(
                            label="Download",
                            data=data,
                            file_name=path_obj.name,
                            mime=mime_type,
                            key=f"download_{path_str}", # Stable key
                            use_container_width=True # Correct parameter
                        )
                    except FileNotFoundError:
                        st.error("File not found.", icon="🚨")