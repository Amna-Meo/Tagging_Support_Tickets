import json
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .preprocessor import clean_text


TAG_TAXONOMY = {
    "billing": "Payment, subscription, invoice, refund, charge, pricing issues",
    "technical": "Bug reports, functionality not working, errors, crashes, software issues",
    "account": "Login, password, access, profile management, account recovery",
    "shipping": "Delivery delays, lost packages, address changes, tracking issues",
    "product": "Product inquiries, features, specifications, compatibility questions",
    "returns": "Return requests, refund status, exchange requests",
    "general": "Other inquiries not fitting above categories",
}

TAG_LIST = list(TAG_TAXONOMY.keys())


class VectorStore:
    def __init__(self, persist_dir: str = "./data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.collection = None

        if CHROMADB_AVAILABLE:
            self._init_chroma()
        else:
            self._load_static_examples()

    def _init_chroma(self):
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name="few_shot_examples",
                metadata={"description": "Few-shot examples for ticket classification"},
            )
        except Exception:
            self._load_static_examples()

    def _load_static_examples(self):
        self.collection = None
        static_path = self.persist_dir.parent / "few_shot_examples.json"
        if static_path.exists():
            with open(static_path) as f:
                self._static_examples = json.load(f)
        else:
            self._static_examples = []

    def add_example(self, text: str, label: str, example_id: Optional[str] = None):
        if self.collection is not None:
            self.collection.add(
                documents=[clean_text(text)],
                metadatas=[{"label": label}],
                ids=[example_id or f"example_{hash(text)}"],
            )
        else:
            self._static_examples.append({"text": text, "label": label})

    def get_similar_examples(self, query: str, n: int = 5) -> list[dict]:
        query = clean_text(query)

        if self.collection is not None:
            try:
                results = self.collection.query(query_texts=[query], n_results=n)
                examples = []
                for i, doc in enumerate(results["documents"][0]):
                    examples.append(
                        {"text": doc, "label": results["metadatas"][0][i]["label"]}
                    )
                return examples
            except Exception:
                pass

        return self._get_static_similar(query, n)

    def _get_static_similar(self, query: str, n: int) -> list[dict]:
        if not hasattr(self, "_static_examples") or not self._static_examples:
            return []
        words = set(query.lower().split())
        scored = []
        for ex in self._static_examples:
            ex_words = set(ex["text"].lower().split())
            score = len(words & ex_words)
            scored.append((score, ex))
        scored.sort(reverse=True)
        return [ex for _, ex in scored[:n]]

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(getattr(self, "_static_examples", []))


_vector_store: Optional[VectorStore] = None


def get_vector_store(persist_dir: str = "./data/vector_store") -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_dir)
    return _vector_store
