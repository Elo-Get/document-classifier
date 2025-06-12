import os
from typing import Dict
from pdf2image import convert_from_path
import numpy as np

class OCRPreprocessor:
    def __init__(self, io_loader, text_extractor, text_cleaner, cache_dir: str):
        self.io_loader = io_loader
        self.text_extractor = text_extractor
        self.text_cleaner = text_cleaner
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def preprocess_and_cache(self, doc_id: int, path: str) -> str:
        cache_path = os.path.join(self.cache_dir, f"{doc_id}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()

        pages = self.io_loader.load_pages(path, auto_rotate=True)
        full_text = ''
        for page in pages:
            raw_text = self.text_extractor.image_to_text(page)
            clean = self.text_cleaner.clean_text(raw_text)
            full_text += clean + ' '

        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(full_text.strip())
        return full_text.strip()
