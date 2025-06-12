import numpy as np
from PIL import Image
import pytesseract

class OCRTextExtractor:
    def __init__(self, lang: str = 'fra+eng'):
        self.lang = lang
        self.custom_config = r'--oem 3 --psm 6'
        if not pytesseract.get_tesseract_version():
            raise EnvironmentError("Tesseract n'est pas installé ou non accessible depuis PATH.")
    
    def image_to_text(self, image: np.ndarray) -> str:
        """
        Convertit une image NumPy (grayscale ou RGB) en texte via Tesseract OCR.

        :param image: Image sous forme de tableau NumPy (uint8)
        :return: Texte OCRisé
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("L'entrée doit être un tableau NumPy.")
        
        text = pytesseract.image_to_string(image, lang=self.lang, config=self.custom_config)
        return text.strip()