import re


def normalize_text(text: str) -> str:
    """Basic text cleanup for embedding quality."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\u200b", "", text)
    return text.strip()

