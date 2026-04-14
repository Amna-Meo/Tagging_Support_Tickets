import re
import unicodedata

MIN_WORD_COUNT = 10
MAX_TOKENS = 2048


def clean_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def count_words(text: str) -> int:
    return len(text.split())


def validate_input(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Text cannot be empty"

    cleaned = clean_text(text)
    word_count = count_words(cleaned)

    if word_count < MIN_WORD_COUNT:
        return False, f"Text too short ({word_count} words, minimum {MIN_WORD_COUNT})"

    return True, ""


def truncate_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])
