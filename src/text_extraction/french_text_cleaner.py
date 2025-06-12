from spellchecker import SpellChecker
from nltk.corpus import stopwords
import nltk
import re
import string
import unicodedata

class FrenchTextCleaner:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.spell = SpellChecker(language='fr')
        self.stop_words = set(stopwords.words('french'))

    def clean_text(self, text: str) -> str:
        text = self._normalize_text(text)
        tokens = text.split()
        corrected = [
            self.spell.correction(token)
            for token in tokens
            if token not in self.stop_words
        ]
        corrected = [word for word in corrected if word is not None]
        return ' '.join(corrected)

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text