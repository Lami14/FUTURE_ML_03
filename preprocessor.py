"""
preprocessor.py
---------------
Text cleaning and preprocessing pipeline for resume and job description text.
Uses NLTK for tokenisation and lemmatisation.
"""

import re
import string
import nltk
import spacy

# Download required NLTK resources on first run
for resource in ["stopwords", "punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Attempt to load spaCy model; fall back gracefully
try:
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    NLP = None
    SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core cleaning helpers
# ---------------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Strip http/https URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    """Strip email addresses."""
    return re.sub(r"\S+@\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    """Strip common phone number patterns."""
    return re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", " ", text)


def remove_special_characters(text: str) -> str:
    """Remove punctuation and non-alphanumeric characters."""
    text = re.sub(r"[^\w\s]", " ", text)          # punctuation → space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # keep only alphanumeric
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Main preprocessing functions
# ---------------------------------------------------------------------------

def basic_clean(text: str) -> str:
    """
    Lightweight cleaning: lowercase, remove noise.
    Preserves multi-word phrases (good for skill matching).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    text = normalise_whitespace(text)
    return text


def deep_clean(text: str) -> str:
    """
    Full NLP pipeline: clean + tokenise + remove stopwords + lemmatise.
    Returns a single string of processed tokens.
    Best for TF-IDF vectorisation.
    """
    text = basic_clean(text)
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


def extract_noun_phrases(text: str) -> list:
    """
    Use spaCy to extract noun phrases from text.
    Useful for identifying skills expressed as multi-word phrases.
    Falls back to simple bigrams if spaCy is unavailable.
    """
    if SPACY_AVAILABLE and NLP is not None:
        doc = NLP(text[:100_000])  # spaCy has a default limit
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    # Fallback: return individual tokens
    tokens = word_tokenize(basic_clean(text))
    return tokens


def preprocess_resume(text: str) -> dict:
    """
    Full preprocessing for a single resume.
    Returns a dict with cleaned variants for different uses.
    """
    return {
        "raw": text,
        "cleaned": basic_clean(text),
        "vectorizable": deep_clean(text),
        "noun_phrases": extract_noun_phrases(basic_clean(text))
    }
