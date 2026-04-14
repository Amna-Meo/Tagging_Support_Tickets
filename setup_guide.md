# Setup Guide

This guide walks through setting up the Auto-Tagging Support Ticket System.

## Prerequisites

- Python 3.10 or higher
- 8GB RAM (minimum)
- ~3GB disk space for models

## Step 1: Clone and Setup Environment

```bash
cd "/home/amna_meo/Auto Tagging Support Tickets"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Install PyTorch (For GPU Acceleration)

### NVIDIA GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only (Default)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Step 3: Download a Model (Automatic)

The first time you run the app, it will automatically download the default model (`microsoft/phi-2`) via HuggingFace Transformers.

For a smaller/faster model, modify `src/classifier.py`:
```python
DEFAULT_MODEL = "gpt2"  # ~500MB, fastest
DEFAULT_MODEL = "microsoft/phi-2"  # ~1.5GB, better quality
```

**Note:** The classifier uses HuggingFace Transformers (not Ollama). It automatically falls back to keyword-based classification if the LLM fails.

## Step 4: Test Without Downloading (Optional)

You can test the keyword-based fallback immediately without any model:

```bash
source .venv/bin/activate
python -c "
from src.classifier import Classifier
c = Classifier()
result = c.classify('I was charged twice for my subscription')
print(result)
"
```

## Step 5: Start the UI

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

Open http://localhost:8501

**Note:** Tickets must have at least 10 words. Shorter text returns `needs_more_info` status.

## Model Options

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `gpt2` | ~500MB | Fastest | Basic |
| `microsoft/phi-2` | ~1.5GB | Fast | Good |
| `microsoft/Phi-3-mini-128k-instruct` | ~2GB | Medium | Better |
| `mistralai/Mistral-7B-Instruct-v0.2` | ~14GB | Slow | Best |

To use a different model, edit `src/classifier.py`:
```python
DEFAULT_MODEL = "gpt2"
```

## GPU Setup (Optional)

If you have an NVIDIA GPU:

```bash
# Install CUDA drivers
nvidia-smi

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

The system automatically detects and uses GPU when available.

## Troubleshooting

### Out of Memory

- Use smaller model: `gpt2`
- CPU mode: models run on CPU automatically if no GPU

### Slow Classification

- Use `gpt2` for fastest results
- Enable GPU for faster inference
- Reduce text length

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

1. Run the app: `streamlit run ui/app.py`
2. Try the keyword-based fallback immediately
3. Go to **Classify** → enter a ticket → click **Classify**
4. Go to **Review Queue** to accept/edit predictions
5. Accumulate corrections in **History**
