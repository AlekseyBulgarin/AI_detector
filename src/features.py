import re

import nltk
import numpy as np
from nltk.corpus import stopwords


try:
	STOP_WORDS = set(stopwords.words("russian"))
except LookupError:
	# Render downloads the resource during the build; keep local setup self-healing.
	nltk.download("stopwords", quiet=True)
	STOP_WORDS = set(stopwords.words("russian"))


def extract_features(text):
    """Extract the five numeric features used by the baseline model."""
    if not text or len(text.strip()) < 20:
        return [0, 0, 0, 0, 0]

    sentences = re.split(r"[.!?]+", text)
    sentences = [sentence for sentence in sentences if sentence.strip()]
    words = re.findall(r"\w+", text.lower())

    if not sentences or not words:
        return [0, 0, 0, 0, 0]

    avg_sent_len = len(words) / len(sentences)
    unique_ratio = len(set(words)) / len(words)
    stopword_ratio = sum(word in STOP_WORDS for word in words) / len(words)
    special_punct = re.findall(r"[—–…]", text)
    punct_ratio = len(special_punct) / len(text)
    word_lengths = [len(word) for word in words]
    std_word_len = np.std(word_lengths) if word_lengths else 0

    return [
        avg_sent_len,
        unique_ratio,
        stopword_ratio,
        punct_ratio,
        std_word_len,
    ]
