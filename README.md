# Multi_modal_LVLM
# 🎬 Video-RAG Pipeline — From Scratch

A full end-to-end implementation of **Video-RAG: Visually-Aligned Retrieval-Augmented Long Video Comprehension** in a single unified notebook.

## 📓 Notebook Workflow

| Section | Description |
|---|---|
| **1. Setup & Environment** | Install packages, configure directories, and initialize shared parameters |
| **2. Feature Extraction** | Download video, sample frames, and run ASR + OCR + Object Detection |
| **3. Indexing & Retrieval** | Build BM25 + FAISS indices and perform hybrid retrieval via RRF |
| **4. LVLM Integration** | Query decoupling, keyframe selection, and augmented generation with Qwen2-VL |

## 🚀 How to Run

### On Google Colab:
1. Upload `video_rag_pipeline.ipynb` to Colab.
2. Set runtime to **GPU (T4 or better)**: `Runtime` → `Change runtime type` → `T4 GPU`.
3. Run all cells sequentially from top to bottom.

### On RunPod:
1. Start a PyTorch pod with CUDA support.
2. Upload `video_rag_pipeline.ipynb`.
3. Launch `jupyter notebook` and execute cells in order.


## 🏗️ Architecture

```
Video + Query
     │
     ├──► Whisper (ASR) ──────────────────────┐
     ├──► EasyOCR (OCR) ─────────────────────►│ Timestamped Text Corpus
     └──► YOLOv8 (Detection) ─────────────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                               BM25 Index          FAISS Index
                                    └──────────┬──────────┘
                                               ▼
                                      Hybrid RRF Retrieval
                                               │
                          ┌────────────────────┘
                          ▼
                    Query Decoupling (LVLM)
                          │
                    Top-K Chunks + Key Frames
                          │
                    Augmented Generation (LVLM)
                          │
                        Answer
```

## ⚙️ Configuration (config.json)

The setup section initializes `video_rag_project/config.json`. You can tune these parameters directly in the notebook:

```json
{
  "whisper_model": "base",          // tiny/base/small/medium/large
  "frame_sample_fps": 1,            // frames per second to extract
  "yolo_model": "yolov8n.pt",       // n/s/m/l/x for speed vs accuracy
  "retrieval_mode": "hybrid",       // bm25 / dense / hybrid
  "top_k_retrieve": 5,              // how many chunks to retrieve
  "lvlm_model": "Qwen/Qwen2-VL-7B-Instruct",
  "max_frames_to_model": 8
}

## 📊 Components

| Component | Library | Notes |
|---|---|---|
| ASR | `openai-whisper` | `base` model ~1GB, `large` ~6GB |
| OCR | `easyocr` | Supports 80+ languages |
| Object Detection | `ultralytics` (YOLOv8) | `yolov8n` = fastest |
| Sparse Retrieval | `rank_bm25` | BM25Okapi |
| Dense Retrieval | `faiss-cpu` + `sentence-transformers` | `all-MiniLM-L6-v2` |
| LVLM | `transformers` | Qwen2-VL-7B (4-bit for T4) |
  
🔍 Pipeline Breakdown
Stage 1 — Setup: Installs all core libraries (Whisper, EasyOCR, YOLOv8, FAISS, Qwen2-VL), configures directories, and saves the central config.json.

Stage 2 — Feature Extraction: Downloads the target video using yt-dlp, extracts frames at 1 fps, and runs Whisper (speech), EasyOCR (on-screen text), and YOLOv8 (objects) to compile a unified, timestamped corpus.json.

Stage 3 — Indexing & Retrieval: Builds a sparse BM25 index and a dense FAISS vector index from the corpus, merging candidate matches via Reciprocal Rank Fusion (RRF).

Stage 4 — LVLM Integration: Executes the 4-stage comprehension pipeline:

Query Decoupling: LVLM refines and breaks down the user prompt for search.
Hybrid Retrieval: Surfaces the top matching timestamped text windows.
Visual Alignment: Pulls key video frames around retrieved timestamps.
Augmented Generation: Qwen2-VL processes both retrieved text and sampled visual frames to generate the grounded response.

