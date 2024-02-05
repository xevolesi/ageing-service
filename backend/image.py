from __future__ import annotations
from typing import TYPE_CHECKING
import base64

import cv2
import torch
import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from fastapi import UploadFile


ACCEPTABLE_IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")
MAX_PIXEL_VALUE = 255
INFERENCE_IMAGE_SIZE = (256, 256)
STYLEGAN_MEANS = [0.5, 0.5, 0.5]
STYLEGAN_STDS = [0.5, 0.5, 0.5]


def preprocess_image(image: NDArray[np.uint8]) -> torch.FloatTensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INFERENCE_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor = image.astype(np.float32) / MAX_PIXEL_VALUE
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor -= np.array(STYLEGAN_MEANS)[:, None, None]
    tensor /= np.array(STYLEGAN_STDS)[:, None, None]
    tensor = np.expand_dims(tensor, axis=0)
    return torch.from_numpy(tensor)


def stylegan_tensor_to_image(stylegan_tensor: torch.FloatTensor) -> NDArray[np.uint8]:
    stylegan_tensor = (
        stylegan_tensor.cpu().detach().squeeze(dim=0).permute(1, 2, 0).numpy()
    )
    stylegan_tensor = (stylegan_tensor + 1) / 2
    stylegan_tensor = np.clip(stylegan_tensor, 0, 1) * 255
    return stylegan_tensor.astype(np.uint8)


def encode_pytorch_image(
    image_tensor: torch.FloatTensor, image_spatial_size: tuple[int, int], extension: str = ".jpg"
) -> bytes:
    image = stylegan_tensor_to_image(image_tensor)
    image = cv2.resize(image, image_spatial_size, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(extension, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1])


def decode_fastapi_file(fastapi_file: UploadFile) -> NDArray[np.uint8]:
    data = np.frombuffer(fastapi_file.file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
