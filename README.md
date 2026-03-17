# AI-Powered Chest X-ray Multi-Label Classification & Impression Generation
**Using Medical Vision and Language Models**

An end-to-end retrieval-augmented generation (RAG) pipeline for automated chest X-ray report generation. The system combines medical vision-language models, multi-label disease classification, and large language models to generate structured radiology reports grounded in evidence from similar cases.

## Key Features

1. **Medical Image Embedding:** Uses BiomedCLIP (ViT-B/16) to encode chest X-rays into 512-dimensional embeddings.
2. **Multi-Label Classification:** Temperature-calibrated neural classifier predicts probabilities for 14 CheXpert disease labels.
3. **Abnormality Scoring:** Unsupervised abnormality detection using cosine distance from a pre-computed healthy training centroid.
4. **Stratified Retrieval with Contrastive Re-Ranking:** FAISS-based retrieval that stratifies cases into abnormal and healthy buckets, re-ranked using a contrastive scoring function to boost clinically relevant matches.
5. **LLM-Based Report Generation:** Generates structured radiology reports (Findings, Comparison, Impression) using Llama-3.1-8B or GPT-4o-mini, heavily grounded in retrieved evidence.

---

## Directory Structure
```
major-project/
├── demo_app.py                # Gradio-based interactive web UI
├── pipeline_config.json       # Pipeline configuration parameters
├── src/                       # Core pipeline source code
│   ├── pipeline_standalone.py # Main MedicalImagingPipeline class
│   ├── evaluate_batch.py      # Batch evaluation loop
│   └── retrain_classifier.py  # Classifier training script
├── scripts/                   # Utility and evaluation scripts
│   ├── run_ablation_check.py  # Runs the 5-mode N=100 ablation study
│   ├── run_judge_eval.py      # LLM-as-Judge qualitative evaluation
│   ├── enrich_faiss_metadata.py
│   └── compute_healthy_embeddings.py
├── notebooks/                 # Jupyter notebooks for exploration/training
├── docs/                      # Detailed project documentation and results
│   ├── PROJECT_EXPERIMENT_INFO.md  # Comprehensive technical details & hyperparameters
│   ├── PROJECT_RESULTS.md          # Full ablation, cross-LLM, and judge results
│   └── PIPELINE_SETUP.md           # Setup instructions
├── models/                    # PyTorch model checkpoints
├── results/                   # Final evaluation JSON outputs
├── embeddings/                # Pre-computed image embeddings (gitignored)
├── faiss_index/               # FAISS vector database (gitignored)
└── image-data/                # MIMIC-CXR dataset (gitignored)
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd major-project
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Copy `.env.example` to `.env` and add your OpenAI API key (if using GPT-4o-mini for report generation).
   ```bash
   cp .env.example .env
   # Edit .env to add OPENAI_API_KEY
   ```

4. **Data Prerequisites:**
   To run the pipeline and demo, you need the pre-computed embeddings, FAISS index, trained models, and the MIMIC-CXR test set images placed in the `image-data/`, `embeddings/`, and `faiss_index/` directories (these are over 80GB and not tracked in Git).

---

## Usage

### 1. Interactive Demo
Launch the Gradio web UI to upload a chest X-ray and generate a report dynamically:
```bash
python demo_app.py
```
This will start a local web server (typically at `http://127.0.0.1:7860`).

### 2. Batch Evaluation
Run the pipeline programmatically on the test set:
```bash
python src/evaluate_batch.py --csv image-data/processed/test.csv --ablation full --sample_size 20
```
Available ablation modes: `full` (proposed), `cosine_only`, `stratify_only`, `contrastive_only`, `no_retrieval`.

### 3. Ablation Study
To reproduce the full N=100 ablation study comparing all 5 modes:
```bash
python scripts/run_ablation_check.py
```

---

## Documentation

For an in-depth explanation of the architecture, datasets, class distributions, hyperparameters, metrics, and quantitative results, please refer to:
- [docs/PROJECT_EXPERIMENT_INFO.md](docs/PROJECT_EXPERIMENT_INFO.md) - Comprehensive Technical Spec
- [docs/PROJECT_RESULTS.md](docs/PROJECT_RESULTS.md) - Detailed Result Tables

---

## Disclaimer
⚕️ For research and educational purposes only. Not for clinical use.
