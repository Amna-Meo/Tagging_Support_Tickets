# Auto-Tagging Support Ticket System (HITL) - Detailed Specification

## 1. Project Overview

**Objective**: Automatically categorize free-text support tickets into the Top 3 most probable tags.
**Primary User**: Customer Support Leads / Triage Agents.
**Deployment**: Localized single-node environment (Offline-capable).

## 2. Tag Taxonomy

Initial tag set (expandable):
- `billing` - Payment, subscription, invoice, refund issues
- `technical` - Bug reports, functionality not working, errors
- `account` - Login, password, access, profile management
- `shipping` - Delivery delays, lost packages, address changes
- `product` - Product inquiries, features, compatibility
- `returns` - Return requests, refund status
- `general` - Other inquiries not fitting above categories

## 3. Multi-Mode Classification

### 3.1 Zero-Shot Mode (Default)
- System prompt contains tag definitions
- Direct LLM inference with no examples
- Fastest response time

### 3.2 Few-Shot Mode
- Retrieve 3-5 semantically similar examples from local vector store (ChromaDB)
- Examples include ticket text → correct tag mapping
- Fallback to static example list if vector store unavailable

### 3.3 Fine-Tuned Mode (Target)
- Quantized model (llama3.2:3b or llama3.1:8b)
- Loaded from `./models/fine_tuned/`
- Trained on correction history exported to JSONL

## 4. Human-in-the-Loop (HITL) & Active Learning

### 4.1 Correction Interface (Streamlit UI)
- Display: ticket text, AI-predicted tags (with confidence), human correction input
- Actions: Accept All, Reject All, Selective override
- Queue-based workflow: Review → Correct → Submit

### 4.2 SQLite Schema (corrections.db)

```sql
CREATE TABLE corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_text TEXT NOT NULL,
    ticket_id TEXT,
    predicted_tags TEXT NOT NULL,  -- JSON array
    predicted_confidences TEXT NOT NULL,  -- JSON array
    accepted_tags TEXT NOT NULL,  -- JSON array
    confidence_delta REAL,  -- avg(human_confidence) - avg(ai_confidence)
    mode TEXT CHECK(mode IN ('zero_shot', 'few_shot', 'fine_tuned')),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    reviewer_id TEXT
);

CREATE INDEX idx_timestamp ON corrections(timestamp);
CREATE INDEX idx_mode ON corrections(mode);
```

### 4.3 JSONL Export Format

```json
{"text": "ticket text", "label": "billing"}
{"text": "ticket text", "label": "technical"}
```

Export trigger: Manual via UI button OR after 100 new corrections (configurable).

## 5. Input/Output Specification

### 5.1 Input Formats

**Single Entry**:
```json
{"text": "I was charged twice for my order"}
```

**CSV Upload**:
| id | text |
|----|------|
| TKT-001 | Ticket text here |

**JSON Upload**:
```json
{"tickets": [{"id": "TKT-001", "text": "..."}]}
```

### 5.2 Output Format

```json
{
  "ticket_id": "TKT-001",
  "tags": [
    {"tag": "billing", "confidence": 0.82},
    {"tag": "refunds", "confidence": 0.12},
    {"tag": "general", "confidence": 0.04}
  ],
  "status": "success",
  "mode": "zero_shot"
}
```

### 5.3 Fallback Logic

| Condition | Response |
|-----------|----------|
| Text < 10 characters (word count) | `needs_more_info` status |
| All confidences < 0.3 | `needs_more_info` status |
| Ollama unavailable | Return error with retry suggestion |

## 6. Technical Implementation

### 6.1 Directory Structure

```
/home/amna_meo/Auto Tagging Support Tickets/
├── spec.md
├── corrections.db
├── data/
│   ├── corrections_export.jsonl
│   └── few_shot_examples.json
├── models/
│   └── fine_tuned/
├── ui/
│   └── app.py  (Streamlit)
├── src/
│   ├── __init__.py
│   ├── classifier.py      # LLM orchestration
│   ├── preprocessor.py    # Text cleaning
│   ├── vector_store.py    # ChromaDB for few-shot
│   ├── database.py        # SQLite operations
│   └── exporter.py       # JSONL export
├── requirements.txt
└── README.md
```

### 6.2 Dependencies

```
ollama>=0.1.0
chromadb>=0.4.0
streamlit>=1.28.0
sqlalchemy>=2.0.0
pandas>=2.0.0
```

### 6.3 Ollama Configuration

- Default model: `llama3.2:3b` (adjustable via config)
- System prompt template defined in `src/classifier.py`
- Connection check on startup; graceful error if unavailable

### 6.4 Performance Targets

| Mode | CPU Latency | GPU Latency |
|------|-------------|-------------|
| Zero-Shot | < 10s | < 3s |
| Few-Shot | < 15s | < 5s |
| Fine-Tuned | < 8s | < 2s |

### 6.5 Text Preprocessing

- Strip leading/trailing whitespace
- Normalize unicode
- Preserve original text in DB (store cleaned version separately)
- Max tokenization: truncate to 2048 tokens

## 7. UI Specification

### 7.1 Pages

1. **Classify**: Single ticket entry or batch upload
2. **Review Queue**: Pending tickets for human review
3. **History**: Past corrections with search/filter
4. **Settings**: Model selection, export, configuration

### 7.2 Review Workflow

```
[Incoming Ticket] → [AI Classification] → [Queue]
                                           ↓
                              [Reviewer: Accept/Reject/Edit]
                                           ↓
                              [Save to corrections.db]
```

### 7.3 Authentication

- None for MVP (single-user local deployment)
- reviewer_id stored as optional field for multi-user future

## 8. Fine-Tuning Pipeline

### 8.1 Training Configuration

- Framework: Ollama fine-tuning (or llamafactory if needed)
- Method: LoRA (low-rank adaptation)
- Parameters:
  - learning_rate: 2e-4
  - epochs: 3-5
  - batch_size: 4
  - quant: 4-bit

### 8.2 Training Trigger

Manual: Export button in UI generates JSONL, then run:
```bash
python -m src.train --data data/corrections_export.jsonl --output models/fine_tuned/
```

## 9. Error Handling

| Scenario | Response |
|----------|----------|
| Ollama not running | Alert in UI + offer to retry |
| Invalid input format | Validation error with expected format |
| Model timeout | Retry once, then return error |
| SQLite write failure | Log error, alert user, do not lose data |

## 10. Acceptance Criteria

1. Single ticket classification returns valid JSON with 1-3 tags
2. Batch CSV processing handles 100+ tickets
3. Corrections are persisted to SQLite
4. JSONL export contains all corrections
5. Streamlit UI loads without errors
6. Zero-shot classification completes in < 15s on CPU
