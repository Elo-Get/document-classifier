import os 

from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
from PIL import Image
import pytesseract

class PDFIo:
    
    def get_greyscale(self, page: np.ndarray) -> np.ndarray:
        """
        Convert a page image to greyscale.
        :param page: The page image as a numpy array.
        :return: The greyscale image as a numpy array.
        """
        if len(page.shape) == 3:
            return np.array(Image.fromarray(page).convert('L'))
        return page

    def detect_rotation_angle(self, image: np.ndarray) -> int:
        """
        Use Tesseract to detect the rotation angle of a page.
        :param image: The image to analyze.
        :return: The angle (in degrees) to rotate clockwise to correct orientation.
        """

        image = Image.fromarray(image)
        image = image.convert('RGB')
        osd = pytesseract.image_to_osd(image, config='--psm 0')
        
        for line in osd.split('\n'):
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
                return angle
        
        return 0  

    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate the image by the given angle.
        :param image: The image to rotate.
        :param angle: The angle in degrees to rotate the image clockwise.
        :return: The rotated image as a numpy array.
        """
        if angle == 0:
            return image

        pil_image = Image.fromarray(image)
        rotated = pil_image.rotate(-angle, expand=True)  
        return np.array(rotated)

    def load_pages(self, path: str, auto_rotate: bool = True) -> np.ndarray:
        """
        Load greyscale pages from a PDF file and optionally auto-rotate them based on detected orientation.
        :param path: The path to the PDF file.
        :param auto_rotate: If True, automatically rotate pages based on detected orientation.
        :return: A numpy array of greyscale images.
        :raises FileNotFoundError: If the specified PDF file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        
        images = convert_from_path(path)
        corrected_images = []

        for img in images:
            np_img = np.array(img)
            if auto_rotate:
                angle = self.detect_rotation_angle(np_img)
                np_img = self.rotate_image(np_img, angle)
            corrected_images.append(np_img)

        return np.array(corrected_images)
