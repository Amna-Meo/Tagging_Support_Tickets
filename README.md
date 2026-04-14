# Auto-Tagging Support Ticket System (HITL)

A local-first, AI-driven classification engine for categorizing support tickets. Zero operational cost, full data privacy.

## Features

- **Multi-Mode Classification**: Zero-shot, Few-shot, and Fine-tuned modes
- **Human-in-the-Loop**: Review and correct AI predictions
- **Active Learning**: Corrections feed into fine-tuning dataset
- **Local LLM**: Runs entirely offline via Hugging Face Transformers
- **Batch Processing**: Single tickets or CSV/JSON uploads
- **Keyword Fallback**: Works immediately without downloading models

## Tags

| Tag | Description |
|-----|-------------|
| `billing` | Payment, subscription, invoice, refund issues |
| `technical` | Bugs, errors, functionality not working |
| `account` | Login, password, access, profile |
| `shipping` | Delivery, lost packages, tracking |
| `product` | Features, specifications, compatibility |
| `returns` | Return requests, refund status |
| `general` | Other inquiries |

## Quick Start

```bash
cd "Auto Tagging Support Tickets"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

Open http://localhost:8501

**Works immediately with keyword-based classification. No model download required.**

## Installation

1. Create virtual environment and install dependencies
2. Optionally install PyTorch with GPU support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. Run: `streamlit run ui/app.py`

## Usage

### Classify Page
- Enter single tickets or upload CSV/JSON
- Choose mode: Zero-shot, Few-shot, or Fine-tuned
- Results go to Review Queue

### Review Queue
- Accept/edit AI predictions
- Corrections saved to SQLite

### History
- View all corrections
- Export to JSONL for fine-tuning

## Output Format

```json
{
  "ticket_id": "TKT-001",
  "tags": [
    {"tag": "billing", "confidence": 0.82},
    {"tag": "refunds", "confidence": 0.12}
  ],
  "status": "success",
  "mode": "zero_shot"
}
```

## Project Structure

```
├── requirements.txt
├── corrections.db            # SQLite (created at runtime)
├── data/
│   ├── few_shot_examples.json
│   └── corrections_export.jsonl
├── models/fine_tuned/        # Fine-tuned models
├── src/
│   ├── preprocessor.py       # Text cleaning
│   ├── database.py           # SQLite operations
│   ├── vector_store.py       # ChromaDB few-shot
│   ├── classifier.py         # LLM orchestration
│   └── exporter.py           # JSONL export
└── ui/app.py                 # Streamlit UI
```

## Models

| Model | Size | Notes |
|-------|------|-------|
| `gpt2` | ~500MB | Fastest, works without GPU |
| `microsoft/phi-2` | ~1.5GB | Default, good balance |
| `mistralai/Mistral-7B` | ~14GB | Best quality, needs GPU |

Change model in `src/classifier.py` → `DEFAULT_MODEL`

## Fallback Logic

- Text < 10 words → `needs_more_info`
- All confidences < 0.3 → `needs_more_info`
- Model unavailable → keyword-based classification

## Requirements

- Python 3.10+
- 8GB RAM
- Optional: NVIDIA GPU with 4GB+ VRAM
