from typing import Optional
import joblib
import numpy as np
from onnxruntime import InferenceSession

class ONNXDocumentClassifier:
    
    def __init__(self, onnx_model_path: str, tfidf_model_path: str):
        self.tfidf = joblib.load(tfidf_model_path)
        self.session = InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, texts: list[str]) -> Optional[list[str]]:
        X_tfidf = self.tfidf.transform(texts).astype(np.float32)
        pred_onx = self.session.run(None, {self.input_name: X_tfidf.toarray()})
        results = []
        
        try:
            for text in pred_onx:
                results.append(text[0])
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            return None
        
        return results
