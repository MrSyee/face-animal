from typing import Dict
import uuid

from PIL import Image
import cv2
import numpy as np
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.post("/image")
def process_image(image):
    image = np.array(Image.open(image.file))
    # preprocess
    cv2.imwrite(f"./data/{str(uuid.uuid4())}.jpg")

    # inference 요청

    # 결과 return
    return {"health": "ok"}
