import re
from typing import Optional
import numpy as np

# matches: word-<newline>word
HYPHEN_LINEBREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)", flags=re.UNICODE)

# optional: only if you really want to normalize slashes
SLASH_JOIN_RE = re.compile(r"(\w+)\s*/\s*(\w+)", flags=re.UNICODE)


def clean_legal_text(text: Optional[str]) -> str:
    """
    Cleans legal text while preserving semantic hyphenation.
    Only removes hyphens caused by line breaks (e.g. PDF artifacts).
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    t = str(text)

    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize tabs
    t = t.replace("\t", " ")

    # ✅ Remove hyphenation ONLY when caused by line breaks
    # Example: "Ther-\nmal" -> "Thermal"
    # Keeps: "Erdwärme-Anlage", "CO2-Preis"
    t = HYPHEN_LINEBREAK_RE.sub(r"\1\2", t)

    # Replace remaining newlines with spaces
    t = re.sub(r"\n+", " ", t)

    # Optional: normalize slashed compounds (use with care)
    # Example: "Wärme-/Kältespeicher" -> "Wärme- und Kältespeicher"
    t = SLASH_JOIN_RE.sub(r"\1 und \2", t)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    return t
