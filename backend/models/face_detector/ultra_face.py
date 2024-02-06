import numpy as np
from numpy.typing import NDArray
from onnxruntime import InferenceSession

from .utils import preprocess_image_detector, predict


class FaceDetector:
    def __init__(self, onnx_path: str) -> None:
        self.__session = InferenceSession(onnx_path)
    
    def __call__(self, image: NDArray[np.uint8], conf_thresh: float = 0.7) -> list[int]:
        tensor = preprocess_image_detector(image)
        input_name = self.__session.get_inputs()[0].name
        confidences, boxes = self.__session.run(None, {input_name: tensor})
        boxes, *_ = predict(image.shape[1], image.shape[0], confidences, boxes, conf_thresh)
        return boxes.tolist()
