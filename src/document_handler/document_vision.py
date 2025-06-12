import numpy as np
from PIL import Image

class DocumentVision:
    
    def display_page(self, page: np.ndarray, title: str = None):
        """
        Display a single page of the document.
        """
        if title:
            print(title)
        img = Image.fromarray(page)
        img.show()