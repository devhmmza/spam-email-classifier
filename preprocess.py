import string
import re

def clean_text(text):
    """
    Lowercase, remove punctuation and digits from the tex.
    """

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text