import os
import time
import shutil
from typing import Optional

from src.document_classifier.classifier_runner import ONNXDocumentClassifier
from src.document_handler.pdf_io import PDFIo
from src.text_extraction.french_text_cleaner import FrenchTextCleaner
from src.text_extraction.ocr_text_extractor import OCRTextExtractor
from src.trainer.document_classifier_classes import DocumentClassifierClasses

import numpy as np

io = PDFIo()
text_cleaner = FrenchTextCleaner()
text_extractor = OCRTextExtractor(lang="fra+eng")


# Constants
DEPOSIT_DIR = "deposit"
DATA_DIR = "data"
CATEGORIES = {
    0: DocumentClassifierClasses.ATTESTATION_HEBERGEMENT.value,
    1: DocumentClassifierClasses.AVIS_IMPOT_TAXE_FONCIERE.value,
    2: DocumentClassifierClasses.AVIS_IMPOT_SUR_REVENUS.value,
    3: DocumentClassifierClasses.BULLETIN_DE_SALAIRE.value,
    4: DocumentClassifierClasses.RELEVE_DE_COMPTE_BANCAIRE.value
}

# Création des dossiers de catégories
for folder in CATEGORIES.values():
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)

# Init du runner modèle
runner = ONNXDocumentClassifier(onnx_model_path="modeles/clf.onnx",
                                 tfidf_model_path="modeles/tfidf.pkl")

def get_next_index(category_folder : str):
    """
    Retourne l'index du dernier pdf +1 dans le dossier category_folder
    :param category_folder: Chemin du dossier de la catégorie
    :return: Index du prochain fichier PDF à créer
    """
    
    files = os.listdir(category_folder)
    pdf_indices = []
    for f in files:
        if f.endswith(".pdf"):
            try:
                idx = int(f.split(".pdf")[0])
                pdf_indices.append(idx)
            except ValueError:
                pass
    return max(pdf_indices, default=0) + 1

def process_pdf(file_path : str) -> Optional[str]:
    """
    Extraction texte, nettoyage, prédiction catégorie
    :param file_path: Chemin du fichier PDF à traiter
    :return: Texte nettoyé ou None en cas d'erreur
    """
    
    try:
        pages = io.load_pages(file_path, auto_rotate=True)
        result_txt = text_extractor.image_to_text(pages[0])
        cleaned_text = text_cleaner.clean_text(result_txt)
        return cleaned_text
    except Exception as e:
        print(f"Erreur extraction fichier {file_path} : {e}")
        return None
    
def get_category_index(category_name: str) -> int:
    """
    Retourne l'index de la catégorie à partir de son nom
    :param category_name: Nom de la catégorie
    :return: Index de la catégorie ou -1 si non trouvée
    """
    
    for idx, name in CATEGORIES.items():
        if name == category_name:
            return idx
    return -1

def predict_category(text: str) -> int:
    """
    Prédiction de la catégorie du texte
    :param text: Texte à classifier
    :return: Index de la catégorie prédite
    """
    
    predictions = runner.predict([text])
    print(f"Prédictions obtenues : {predictions}")
    
    if not predictions or len(predictions) == 0:
        print("Aucune prédiction obtenue")
        return -1
        
    try:
        cat_idx = get_category_index(predictions[0])
    except Exception as e:
        cat_idx = -1
    return cat_idx

def move_file_to_category(file_path : str, category_idx: int):
    """
    Déplace le fichier vers le dossier de la catégorie correspondante
    :param file_path: Chemin du fichier à déplacer
    :param category_idx: Index de la catégorie
    """
    
    category_folder = os.path.join(DATA_DIR, CATEGORIES[category_idx])
    next_idx = get_next_index(category_folder)
    new_filename = f"{next_idx}.pdf"
    dest_path = os.path.join(category_folder, new_filename)
    shutil.move(file_path, dest_path)
    print(f"Fichier déplacé vers {dest_path}")
    
def remove_file(file_path: str, reason: str):
    """
    Supprime le fichier et affiche un message de raison
    :param file_path: Chemin du fichier à supprimer
    :param reason: Raison de la suppression
    """
    
    print(f"Suppression fichier {file_path} - {reason}")
    os.remove(file_path)

def main_loop(refresh_interval: int = 5):
    processed_files = set()
    while True:
        files = os.listdir(DEPOSIT_DIR)
        for f in files:
            full_path = os.path.join(DEPOSIT_DIR, f)
            if full_path in processed_files:
                continue  
            
            if os.path.isfile(full_path):
                if f.lower().endswith(".pdf"):
                    print(f"Traitement fichier {f} ...")
                    text = process_pdf(full_path)
                    if isinstance(text, str) and len(text) > 0:
                        cat_idx = predict_category(text)
                        if cat_idx == -1:
                            remove_file(full_path, "catégorie non trouvée")
                            continue
                        print(f"Catégorie prédite : {CATEGORIES[cat_idx]}")
                        move_file_to_category(full_path, cat_idx)
                    else:
                        remove_file(full_path, "texte vide ou erreur d'extraction")
                else:
                    remove_file(full_path, "fichier non PDF")
            
            processed_files.add(full_path)
        
        time.sleep(refresh_interval) 

if __name__ == "__main__":
    main_loop(2)
