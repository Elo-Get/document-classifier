import onnxruntime as ort
import numpy as np

class ONNXDocumentClassifier:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def predict(self, texts: list[str]) -> list[str]:
        inputs = np.array(texts).reshape(-1, 1).astype(np.object_)
        outputs = self.session.run([self.label_name], {self.input_name: inputs})
        return outputs[0]  
