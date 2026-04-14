import json
import os
import re
from typing import Optional

from .preprocessor import clean_text, validate_input, truncate_to_tokens

HF_TOKEN = os.getenv("HF_TOKEN")
from .vector_store import TAG_TAXONOMY, TAG_LIST, get_vector_store


DEFAULT_MODEL = "microsoft/phi-2"
MIN_CONFIDENCE = 0.3
TOP_K = 3

SYSTEM_PROMPT = """You are a support ticket classifier. Classify the ticket into one of these categories:
- billing: Payment, subscription, invoice, refund issues
- technical: Bugs, errors, functionality not working
- account: Login, password, access, profile
- shipping: Delivery, lost packages, tracking
- product: Features, specifications, compatibility
- returns: Return requests, refund status
- general: Other inquiries

Return ONLY valid JSON with this structure:
{{"tags": [{{"tag": "category", "confidence": 0.XX}}], "reasoning": "brief"}}

Rules:
- Return top 3 most likely categories
- Confidences sum to 1.0
- Use lowercase categories
"""


class Classifier:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        mode: str = "zero_shot",
        force_fallback: bool = False,
    ):
        self.model_name = model
        self.mode = mode
        self.force_fallback = force_fallback
        self.vector_store = get_vector_store()
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, token=HF_TOKEN, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=HF_TOKEN,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                )
                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                )
            except Exception as e:
                self._pipeline = None

    def _build_prompt(self, ticket_text: str) -> str:
        prompt = SYSTEM_PROMPT + f"\n\nTicket: {ticket_text}\n\nResponse:"

        if self.mode == "few_shot":
            examples = self.vector_store.get_similar_examples(ticket_text, n=3)
            if examples:
                prompt += "\n\nExamples:\n"
                for ex in examples:
                    prompt += f'- "{ex["text"][:100]}..." -> {ex["label"]}\n'

        return prompt

    def _parse_response(self, text: str) -> Optional[dict]:
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def classify(
        self,
        ticket_text: str,
        ticket_id: Optional[str] = None,
    ) -> dict:
        is_valid, error_msg = validate_input(ticket_text)
        if not is_valid:
            return {
                "ticket_id": ticket_id,
                "tags": [],
                "status": "needs_more_info",
                "reason": error_msg,
                "mode": self.mode,
            }

        cleaned = clean_text(ticket_text)
        truncated = truncate_to_tokens(cleaned)

        if self.force_fallback:
            return self._fallback_classify(ticket_text, ticket_id)

        try:
            self._load_pipeline()
            if self._pipeline:
                prompt = self._build_prompt(truncated)
                result = self._pipeline(
                    prompt, pad_token_id=self._pipeline.tokenizer.eos_token_id
                )
                generated = result[0]["generated_text"]
                parsed = self._parse_response(generated)

                if parsed and "tags" in parsed:
                    tags = parsed["tags"][:TOP_K]
                    confidences = [t["confidence"] for t in tags]

                    if all(c < MIN_CONFIDENCE for c in confidences):
                        return {
                            "ticket_id": ticket_id,
                            "tags": [],
                            "status": "needs_more_info",
                            "reason": "All confidences below threshold",
                            "mode": self.mode,
                        }

                    return {
                        "ticket_id": ticket_id,
                        "tags": tags,
                        "status": "success",
                        "reasoning": parsed.get("reasoning", ""),
                        "mode": self.mode,
                    }
        except Exception:
            pass

        return self._fallback_classify(ticket_text, ticket_id)

    def _fallback_classify(
        self, ticket_text: str, ticket_id: Optional[str], error: Optional[str] = None
    ) -> dict:
        text_lower = ticket_text.lower()
        scores = {}

        keywords = {
            "billing": [
                "charge",
                "payment",
                "invoice",
                "refund",
                "subscription",
                "price",
                "bill",
                "paid",
                "cost",
                "credit",
            ],
            "technical": [
                "error",
                "bug",
                "crash",
                "broken",
                "issue",
                "problem",
                "fix",
                "not working",
                "failed",
            ],
            "account": [
                "login",
                "password",
                "account",
                "access",
                "profile",
                "email",
                "username",
                "register",
            ],
            "shipping": [
                "delivery",
                "shipping",
                "package",
                "tracking",
                "delayed",
                "lost",
                "arrived",
                "carrier",
            ],
            "product": [
                "feature",
                "product",
                "spec",
                "specification",
                "compatible",
                "how to",
                "work with",
            ],
            "returns": ["return", "exchange", "refund status", "send back"],
            "general": [],
        }

        for tag, words in keywords.items():
            if words:
                scores[tag] = sum(1 for w in words if w in text_lower) / len(words)
            else:
                scores[tag] = 0.0

        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        total = sum(s[1] for s in sorted_tags) or 1

        top_tags = [
            {"tag": t[0], "confidence": round(max(t[1], 0.05), 2)}
            for t in sorted_tags[:TOP_K]
        ]

        norm = sum(tag["confidence"] for tag in top_tags)
        if norm > 0:
            for tag in top_tags:
                tag["confidence"] = round(tag["confidence"] / norm, 2)

        return {
            "ticket_id": ticket_id,
            "tags": top_tags,
            "status": "fallback",
            "reason": "Keyword-based classification",
            "mode": self.mode,
        }

    def check_health(self) -> dict:
        try:
            import torch

            if torch.cuda.is_available():
                return {"status": "ready", "device": "cuda", "model": self.model_name}
            return {"status": "ready", "device": "cpu", "model": self.model_name}
        except Exception as e:
            return {"status": "error", "reason": str(e)}


def classify_ticket(
    text: str,
    mode: str = "zero_shot",
    model: str = DEFAULT_MODEL,
    ticket_id: Optional[str] = None,
) -> dict:
    classifier = Classifier(model=model, mode=mode)
    return classifier.classify(text, ticket_id)


import torch
