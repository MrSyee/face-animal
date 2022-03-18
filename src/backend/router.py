import uuid
from typing import Dict

import cv2
import numpy as np
from backend.model import ImageResponse
from fastapi import APIRouter, File, UploadFile
from PIL import Image

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.post("/image", response_model=ImageResponse)
def process_image(image: UploadFile = File(...)):
    id_ = str(uuid.uuid4())
    image = np.array(Image.open(image.file))
    # preprocess
    resized_image = cv2.resize(image, (128, 128))

    # inference 요청

    # 결과 return
    response = {
        "id": id_,
        "inference_result": "cat",
    }
    return response
