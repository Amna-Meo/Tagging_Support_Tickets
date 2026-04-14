# AGENTS.md

## Quick Start
```bash
cd "Auto Tagging Support Tickets"
source .venv/bin/activate
streamlit run ui/app.py
```

## Key Commands
- **Run app**: `streamlit run ui/app.py`
- **Install deps**: `pip install -r requirements.txt`
- **Test without model**: `python -c "from src.classifier import Classifier; print(Classifier().classify('I was charged twice'))"`

## Architecture
- **Entry point**: `ui/app.py` (Streamlit)
- **Core modules**: `src/classifier.py`, `src/database.py`, `src/vector_store.py`, `src/preprocessor.py`, `src/exporter.py`
- **Database**: SQLite at `corrections.db` (auto-created)
- **Data files**: `data/few_shot_examples.json`, `data/corrections_export.jsonl`

## Tags
`billing`, `technical`, `account`, `shipping`, `product`, `returns`, `general`

## Classification Modes
1. **zero_shot** (default): System prompt with tag definitions
2. **few_shot**: Uses ChromaDB vector store with examples from `data/few_shot_examples.json`
3. **fine_tuned**: Loads quantized model from `models/fine_tuned/`

## Models
- Default: `microsoft/phi-2` (~1.5GB)
- Fallback keyword classifier works immediately without any model download
- Change model in `src/classifier.py` → `DEFAULT_MODEL`

## Important Quirks
- Input validation: minimum 10 words (not characters)
- All confidences < 0.3 → `needs_more_info` status
- Classifier falls back to keyword-based if LLM fails
- Spec mentions Ollama but code uses Hugging Face Transformers

## Dependencies
`transformers`, `torch`, `chromadb`, `streamlit`, `sqlalchemy`, `pandas`, `accelerate`

## File Paths
All paths are relative to project root. App creates `corrections.db` at runtime.
