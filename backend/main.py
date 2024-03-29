from collections import defaultdict

import cv2
import numpy as np
import imagehash
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, status

from dao import GenerationDAO
from image import (
    decode_fastapi_file,
    preprocess_image,
    encode_pytorch_image,
    ACCEPTABLE_IMAGE_EXTENSIONS,
)
from inference import model, add_aging_channel, run_on_image
from models.face_detector import FaceDetector

__FD_CHECKPOINT_PATH = "backend/weights/face_detector_optimized.onnx"


app = FastAPI()
face_detector = FaceDetector(__FD_CHECKPOINT_PATH)
__poorman_cache = defaultdict(dict)


@app.post("/predict/", response_model=GenerationDAO)
def predict(ages: list[int], file: UploadFile) -> GenerationDAO:
    try:
        _, ext = file.filename.split(".")
    except ValueError:
        message = f"Expected image file to have one of {ACCEPTABLE_IMAGE_EXTENSIONS} extension, but found none of them"
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=message)
    if ext not in ACCEPTABLE_IMAGE_EXTENSIONS:
        message = f"Expected image extension to be one of {ACCEPTABLE_IMAGE_EXTENSIONS}, but got {ext}"
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=message)

    try:
        image = decode_fastapi_file(file)
    except cv2.error:
        message = (
            "Your image file is corrupted or you are trying to use not an image file."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message
        )

    age_list_done = []
    generated_images_done = []
    image_hash = imagehash.average_hash(Image.fromarray(image))
    if image_hash in __poorman_cache:
        for age in ages:
            if age in __poorman_cache[image_hash]:
                age_list_done.append(age)
                generated_images_done.append(__poorman_cache[image_hash][age])

    age_list = [age for age in ages if age not in age_list_done]
    if not age_list:
        return GenerationDAO(ages=age_list_done, images=generated_images_done)

    # Run generation only for thoose ages that are not in cache.
    top_left_x, top_left_y, bot_right_x, bot_right_y = face_detector(image)[0] 
    image = np.ascontiguousarray(image[top_left_y: bot_right_y, top_left_x: bot_right_x])
    src_image_height, src_image_width = image.shape[:2]
    image_tensor = preprocess_image(image)
    processed_batch = run_on_image(model, add_aging_channel(image_tensor, age_list))
    images = [
        encode_pytorch_image(image_tensor, (src_image_width, src_image_height),  "." + ext)
        for image_tensor in processed_batch
    ]

    # Add newly generated images into done list.
    age_list_done.extend(age_list)
    generated_images_done.extend(images)

    # Update cache with newly generated images.
    for age, generated_image in zip(age_list, images):
        __poorman_cache[image_hash].update({age: generated_image})

    return GenerationDAO(ages=age_list_done, images=generated_images_done)
